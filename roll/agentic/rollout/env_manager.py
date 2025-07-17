import copy
import re
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from threading import Thread, Lock
from typing import Dict, List, Optional, Union, Tuple

import PIL
import numpy as np
import random
import ray
import torch
from ray.util.queue import Queue, Empty
from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizer, ProcessorMixin

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from roll.distributed.scheduler.generate_scheduler import GlobalCounter, RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import pad_to_length
from roll.utils.logging import get_logger

"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/llm_agent/es_manager.py
"""


@dataclass
class EnvStatus:
    """Status of an environment"""

    truncated: bool = False  # done but not success
    terminated: bool = False  # done and success
    num_actions: int = 0  # current action step (single action)
    rewards: List[float] = field(default_factory=list)  # rewards for each turn
    seed: Optional[int] = None  # what seed is used to reset this environment
    step: int = 0  # current step (single step)

    @property
    def done(self):
        return self.truncated or self.terminated


def get_masks_and_scores(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]] = None,
    use_turn_scores: bool = False,
):
    """
    input_ids: shape (bsz, seq_len)
    all_scores: list[list[float], 存储每个env每轮的reward
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    NOTE: important! input_ids is left pad
    """
    # TODO: special tokens add to config
    assistant_turn_start_tokens = tokenizer.encode("<|im_start|>assistant\n")
    turn_start_token = assistant_turn_start_tokens[0]
    turn_starts = torch.where(input_ids == turn_start_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)

    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)  # only learns all assistant turns
    non_prompt_mask = turn_indicators > 2  # learns everything after system prompt + user prompts

    # turn text: '<|im_start|>assistant\n<answer>Right</answer><|im_end|>'
    # <|im_start|>assistant\n 应该mask掉才对，保留<|im_end|>
    for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
        """
        system, user, assistant, user, assistant, user, assistant
        1,2,3,3,4,5,6
        assistant位于第3,5,7...
        """
        turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
        turn_start_position = (input_ids == turn_start_token) & (turn_indicators == turn_indicator)
        batch_size, seq_len = input_ids.shape
        num_tokens = len(assistant_turn_start_tokens)
        turn_start_indices = turn_start_position.nonzero(as_tuple=True)
        mask_matrix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        for batch_idx, start_idx in zip(turn_start_indices[0], turn_start_indices[1]):
            end_idx = start_idx + num_tokens
            if end_idx <= seq_len:
                mask_matrix[batch_idx, start_idx:end_idx] = True
        response_mask[mask_matrix] = False
        if idx == 0:
            non_prompt_mask[mask_matrix] = False

    # TODO: special tokens add to config
    reward_token = tokenizer.encode("<|im_end|>")[0]
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)

    return non_prompt_mask, score_tensor, response_mask


class EnvManager:
    def __init__(
        self,
        worker_config: EnvManagerConfig,
        pipeline_config: AgenticConfig,
        env_config: Dict,
        tokenizer: PreTrainedTokenizer,
        generate_scheduler,
        input_queue: Queue,
        output_queue: Queue,
        thread_lock: Lock,
        processor: Optional[ProcessorMixin] = None,
        collator: Optional[callable] = None,
        mode="train",
    ):
        """
        1. 一个EnvManager持有一个env实例: 执行env.reset, env.step, 管理rollout的状态
            group trajectory表达: group内的init state一致，依赖env_config 中的seed来控制, 一个group内env 对应episode的seed一致
            EnvWorker持有多个EnvManager，通过线程的方式实现EnvWorker内部并行
        2. run_rollout_loop, 持续rollout trajectory, 将done的trajectory回传到output_queue
        TODO:
            1. special tokens add to config
            2. ray max_concurrency 描述多线程是否会更好？
        """
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: Dict = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor: ProcessorMixin = processor
        self.collator = collator
        self.env_entry = None
        self.output_queue = output_queue
        self.input_queue = input_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler
        self.rollout_cache = None
        self.group_seed = None
        self.episode_id = 0
        self.process_input_queue_thread = None
        self.running = False
        self.use_thread_lock = self.env_config.get(
            "use_thread_lock", True
        )  # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()

        self.env_entry = copy.deepcopy(self.env_config)
        self.env_entry["env"] = REGISTERED_ENVS[self.env_entry["env_class"]](self.env_entry["config"])
        self.env_entry["status"] = EnvStatus()

        self.request_counter = GlobalCounter.options(
            name=f"EnvManagerRequestCounter",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        self.request_id: Optional[str] = None

        # Template modification to render thinking state in previous rounds...
        if self.tokenizer:
            self.tokenizer.chat_template = self.tokenizer.chat_template.replace("> ns.last_query_index", "> 0")
        if self.processor:
            self.processor.chat_template = self.processor.chat_template.replace("> ns.last_query_index", "> 0")

    def reset(self):
        entry = self.env_entry
        self.rollout_cache = {
            "env_id": entry["env_id"],
            "history": [],
            "group_id": entry["group_id"],
            "tag": entry["tag"],
            "penalty": 0,
            "frames": [],
        }

        seed = self.group_seed + self.episode_id

        with self.thread_lock:
            entry["env"].reset(seed=seed)

        entry["status"] = EnvStatus(seed=seed)
        next_state = self._handle_mm_state(entry["env"].render())

        # update rollout cache
        self.rollout_cache["history"] = self._update_cache_history(
            self.rollout_cache["history"],
            next_state=next_state,
            legal_actions=entry["env"].get_all_actions(),
            actions_left=entry["max_actions_per_traj"],
            num_actions_info=None,
        )
        self.episode_id += 1
        return self.rollout_cache

    def compute_length_penalty(self, token_length: int) -> float:
        # (tzy) according to orignal design in kimi-1.5, it is better to use a batch of samples to compute min/max length.
        # however, we don't have a batch of samples. moreover, we just want to penalize the length of the response to avoid overthinking.
        # reward = 0.5 - (token_length - min_len) / (max_len - min_len)

        # for response in format: `<answer>X({i},{j})</answer><|im_end|>`(0<=i,j<3), the minimum length is 11.
        min_len = 11
        max_len = 2000
        reward = 0.0
        # penalize the length of the response
        reward = 1 - (token_length - min_len) / (max_len - min_len)
        reward = min(0, reward)  # we also don't want to reward shorter responses.

        if reward < 0:
            reward *= 0.5  # scale to make it comparable with the winning rewards

        return reward

    def step(self, llm_output: DataProto):
        env_input: Dict = self.get_env_input(llm_output)

        entry = self.env_entry
        actions_left_before = entry["max_actions_per_traj"] - entry["status"].num_actions

        # execute actions in env
        valid_actions, lose_for_wrong_format = self._extract_map_valid_actions(entry, env_input["actions"])
        if lose_for_wrong_format:
            _, acc_reward, turn_done, turn_info = entry["env"].get_losing_state()
            executed_actions = []
        else:
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(
                entry["env"], valid_actions[:actions_left_before]
            )
            acc_reward -= self.worker_config.format_penalty

        acc_reward += self.compute_length_penalty(env_input["token_length"])

        status, history = self._log_env_state(
            entry["status"],
            self.rollout_cache["history"],
            entry["env"].render(),
            entry["env"].get_all_actions(),
            entry["max_actions_per_traj"],
            executed_actions,
            valid_actions,
            acc_reward,
            turn_done,
            turn_info,
            env_input,
        )
        status.step += 1
        entry["status"] = status

        max_steps_per_traj = entry.get("max_steps_per_traj", entry["max_actions_per_traj"])
        if status.step >= max_steps_per_traj and not turn_done:
            entry["status"].truncated = True
            entry["status"].terminated = True

        self.rollout_cache["history"] = history

        if self.mode == "val":
            frame = entry["env"].render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                self.rollout_cache["frames"].append(frame)

        return status

    def generate(self, env_output: Dict):
        lm_input: DataProto = self.get_lm_input(env_output, prepare_for_update=False)

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(
            generation_config["max_new_tokens"],
            max(
                self.pipeline_config.sequence_length
                - lm_input.batch["input_ids"].shape[1]
                - generation_config["max_new_tokens"],
                1,
            ),
        )
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                f"maybe you should increase the response_length"
            )
            return None

        gen_batch = lm_input.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=(["multi_modal_data"] if "multi_modal_data" in lm_input.non_tensor_batch else []),
        )
        gen_batch.meta_info["generation_config"] = generation_config
        gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
        self.request_id = str(ray.get(self.request_counter.get_value.remote()))
        gen_batch.meta_info["request_id"] = self.request_id
        gen_batch.meta_info["src_rank"] = self.env_config["env_id"]
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=gen_batch))

        if lm_output is not None:
            # 未被abort
            gen_batch.meta_info.pop("generation_config")
            lm_input = lm_input.repeat(repeat_times=generation_config["num_return_sequences"])
            lm_output.union(lm_input)
        return lm_output

    def run_rollout_loop(self, data: DataProto):
        """
        1. 每次调用run_rollout_loop,
            会持续的play episode, 直到收到采集完成的command
            需要重置seed, 确保每个group的seed一致
            episode_id 置0
        seed更新逻辑:
            group_seed = seed + group_seed
            episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """

        self.start_input_queue_process()
        self.running = True
        self.episode_id = 0

        self.group_seed = data.meta_info["seed"] + self.env_entry["group_seed"]
        env_output = self.reset()

        while self.running:
            lm_output: DataProto = self.generate(env_output)

            status = EnvStatus(truncated=True, terminated=True)
            if lm_output is not None:
                status: EnvStatus = self.step(lm_output)

            if status.done and self.running:
                rollout: DataProto = self.formulate_rollouts()
                traj_group_id = f"{self.env_entry['group_id']}_{self.episode_id}_{self.group_seed}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id], dtype=object)
                self.output_queue.put_nowait(rollout)
                self.rollout_cache = None
                if self.episode_id >= self.worker_config.max_traj_per_env:
                    self.logger.debug(
                        f"env_id: {self.env_config['env_id']} max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop"
                    )
                    break
                env_output = self.reset()

        self.process_input_queue_thread.join()

    def get_lm_input(self, env_output, prepare_for_update: bool) -> DataProto:
        """"""
        llm_input_texts, messages_list = self._format_messages(
            env_output=env_output, prepare_for_update=prepare_for_update, use_raw_llm_response=False
        )
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )  # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        llm_inputs.non_tensor_batch.update(
            {
                "env_ids": np.array([env_output["env_id"]], dtype=object),
                "group_ids": np.array([env_output["group_id"]], dtype=object),
                "llm_input_texts": np.array(llm_input_texts, dtype=object),
                "messages_list": np.array(messages_list, dtype=object),
                "tags": np.array([env_output["tag"]], dtype=object),
            }
        )
        return llm_inputs

    def get_env_input(self, lm_output: DataProto) -> Dict:
        if lm_output.batch is not None and "responses" in lm_output.batch.keys():
            responses = self.tokenizer.batch_decode(lm_output.batch["responses"], skip_special_tokens=True)
            token_lengths = list(map(len, lm_output.batch["responses"]))
        else:  # dataproto has textual responses
            responses = lm_output.non_tensor_batch["response_texts"]
            token_lengths = list(map(lambda x: len(self.tokenizer.encode(x)) + 1, responses))  # + 1 for eos token

        responses = [
            "<think>\n" + response if self.pipeline_config.enable_think else "<answer>" + response
            for response in responses
        ]  # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_output.non_tensor_batch["env_ids"]
        env_id = env_ids[0]
        response = responses[0]
        token_length = token_lengths[0]
        llm_response, actions = self._parse_response(response)
        env_input = {
            "env_id": env_id,
            "llm_raw_response": response,
            "llm_response": llm_response,
            "actions": actions,
            # (tzy) use the token length information to compute length penalty reward.
            "token_length": token_length,
        }
        return env_input

    def formulate_rollouts(self):
        """
        1. 每个env的trajectory 应该是一个rollout
        2. 每个rollout 应该是一个List[Dict]
        3. 每个Dict 应该是一个step的信息
        """
        # print("rollout cache: ", self.rollout_cache)
        llm_input_texts, messages_list = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=False
        )
        # print("llm_input_texts: ", llm_input_texts)
        # print("messages_list: ", messages_list)
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        scores = [[i["reward"] for i in self.rollout_cache["history"]]]
        episode_scores = [sum(i) for i in scores]
        penalty = self.rollout_cache["penalty"]

        non_prompt_mask, score_tensor, response_mask = get_masks_and_scores(
            input_ids, self.tokenizer, scores, use_turn_scores=self.pipeline_config.use_turn_scores
        )
        non_prompt_mask = torch.logical_and(non_prompt_mask, attention_mask)
        response_mask = torch.logical_and(response_mask, attention_mask)

        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        non_prompt_mask = pad_to_length(non_prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        llm_inputs.batch.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "penalty": torch.Tensor([penalty]),
            }
        )
        llm_inputs.non_tensor_batch.update(
            {
                "env_ids": np.array([self.rollout_cache["env_id"]], dtype=object),
                "group_ids": np.array([self.rollout_cache["group_id"]], dtype=object),
                "messages_list": np.array(messages_list, dtype=object),
                "tags": np.array([self.rollout_cache["tag"]], dtype=object),
                "frames": np.array([self.rollout_cache["frames"]], dtype=object),
            }
        )
        # pad to response length
        llm_inputs.batch["llm_response_mask"] = response_mask
        llm_inputs.batch["non_prompt_mask"] = non_prompt_mask
        llm_inputs.batch["response_mask"] = non_prompt_mask
        if self.pipeline_config.enable_response_mask:
            # 只使用llm的response mask，不包含环境的state
            llm_inputs.batch["response_mask"] = response_mask
        first_true_indices = non_prompt_mask.int().argmax(dim=1)
        no_true_mask = ~non_prompt_mask.any(dim=1)
        first_true_indices[no_true_mask] = non_prompt_mask.size(1)
        batch_size, seq_len = non_prompt_mask.size()
        arange = torch.arange(seq_len, device=non_prompt_mask.device).unsqueeze(0).expand(batch_size, -1)
        prompt_mask = arange < first_true_indices.unsqueeze(1)
        llm_inputs.batch["prompt_mask"] = prompt_mask
        llm_inputs.batch["scores"] = score_tensor
        # for llm raw response
        llm_raw_text_list, _ = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=False
        )
        # print("llm_raw_text_list: ", llm_raw_text_list)
        llm_inputs.non_tensor_batch["turn_scores"] = np.array(scores, dtype=object)
        llm_inputs.non_tensor_batch["episode_scores"] = np.array(episode_scores, dtype=object)
        llm_inputs.non_tensor_batch["llm_raw_text_list"] = np.array(llm_raw_text_list, dtype=object)

        entry = self.env_entry
        status = entry["status"]
        env_metric = {
            "success": float(status.terminated and (not status.truncated)),
            "num_actions": status.num_actions,
        }
        custom_metric = {}

        for turn in self.rollout_cache["history"]:
            for k, v in turn.get("info", {}).items():
                if k == "success":
                    env_metric[k] = float(v)
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            # env_metric[k] = np.sum(v) / len(self.rollout_cache['history'])
            env_metric[k] = np.sum(v)

        if self.rollout_cache["history"]:
            self.rollout_cache["history"][-1]["metrics"] = custom_metric

        env_metric = {f"env/{entry['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        self.rollout_cache["metrics"] = env_metric
        llm_inputs.meta_info = {"metrics": env_metric}
        return llm_inputs

    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment"""
        if isinstance(state, str):  # text state
            return state
        elif isinstance(
            state, np.ndarray
        ):  # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode="RGB") for _state in state]
        return results

    def _update_cache_history(
        self, history: List[Dict], next_state, legal_actions, actions_left, num_actions_info: Optional[Dict] = None
    ):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None:  # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)

        entry = {}  # append state to history
        entry["state"] = next_state
        entry["actions_left"] = actions_left
        entry["legal_actions"] = legal_actions
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry["env"].config, "action_lookup", None)
        if action_lookup is None:
            mapped_actions = actions
        else:  # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]

        legal_actions = entry["env"].get_all_actions()
        mapped_actions = [action for action in mapped_actions if action in legal_actions.values()]
        illegal_actions = [action for action in actions if action not in legal_actions.values()]
        lose_for_wrong_format = False
        if len(mapped_actions) != 1 or len(illegal_actions) > 0:
            lose_for_wrong_format = True
            # print(f"Invalid actions: {actions}, mapped actions: {mapped_actions}, legal actions: {legal_actions}")
        return mapped_actions, lose_for_wrong_format

    def _execute_actions(self, env, actions):
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        for action in actions:
            with self.thread_lock:
                _, reward, done, info = env.step(action)
            acc_reward += reward
            turn_info.update(info)  # NOTE: currently use last info for multi-action
            executed_actions.append(action)
            if done:
                turn_done = True
                break
        return acc_reward, turn_info, turn_done, executed_actions

    def _log_env_state(
        self,
        status,
        history,
        cur_obs,
        legal_actions,
        max_actions_per_traj,
        executed_actions,
        all_actions,
        acc_reward,
        turn_done,
        turn_info,
        env_input,
    ) -> Tuple[EnvStatus, List[Dict]]:
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward)
        actions_left = max_actions_per_traj - status.num_actions
        if turn_done:
            status.terminated = True
            status.truncated = not turn_info.get("success", False)
        history = self._update_cache_history(
            history,
            next_state=obs,
            legal_actions=legal_actions,
            actions_left=actions_left,
            num_actions_info={
                "actions": executed_actions,
                "reward": acc_reward,
                "info": turn_info,
                "llm_response": env_input["llm_response"],
                "llm_raw_response": env_input["llm_raw_response"],
            },
        )
        return status, history

    def _format_messages(self, env_output: Dict, prepare_for_update: bool, use_raw_llm_response: bool):
        if "state" in env_output["history"][-1] and (not use_raw_llm_response and prepare_for_update):
            # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            # print(f"removing last state from history: {env_output['history'][-1]}")
            env_output["history"] = env_output["history"][:-1]

        messages = [
            {
                "role": "system",
                "content": self.env_entry["env"].get_prompt(mode="prefix", think=self.pipeline_config.enable_think)[
                    "system"
                ],
            },
            {
                "role": "user",
                "content": self.env_entry["env"].get_prompt(mode="prefix", think=self.pipeline_config.enable_think)[
                    "user"
                ],
            },
        ]

        for idx, content in enumerate(env_output["history"]):
            if messages[-1]["role"] != "user":
                # ensure the last message is user message.
                messages.append({"role": "user", "content": ""})

            turn_idx_content = f"Information of Turn-{idx * 2 + 1}:\n\n"
            messages[-1]["content"] += turn_idx_content
            if "state" in content:
                messages[-1]["content"] += (
                    f"GAME STATE:\n{content['state']}\n\n"
                    f"LEGAL ACTIONS:\n{', '.join(content['legal_actions'].values())}.\n\n"
                    f"You are `X` and your opponent is `O`. Now this is your turn, please take an action from the legal actions above.\n"
                    "Note that you should output your action in the format of `<answer><X({i},{j})></answer>` where `X` is your action and `(i,j)` is the coordinate of the action.\n\n"
                )
            if "llm_raw_response" in content:
                #       改成actions合理吗？
                messages.append(
                    {
                        "role": "assistant",
                        "content": content["llm_response"] if not use_raw_llm_response else content["llm_raw_response"],
                        # "content": content["llm_raw_response"],
                    }
                )
            if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                # reward_content = f"Reward:\n{content['reward']}\n"
                reward_content = ""
                messages.append({"role": "user", "content": reward_content})

        # NOTE: this assertion is important for loss mask computation
        assert all(msg["role"] == "assistant" for msg in messages[2::2])

        if self.processor:
            # processor.chat_template might be different with tokenizer
            # can also set tokenizer.chat_template to processor.chat_template
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=(not prepare_for_update), tokenize=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=(not prepare_for_update), tokenize=False
            )
        if use_raw_llm_response:
            prompt_messages = messages[:2]
            if self.processor:
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages, add_generation_prompt=False, tokenize=False
                )
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, add_generation_prompt=False, tokenize=False
                )
            text = text[len(prompt_text) :]
        if not prepare_for_update:
            if self.pipeline_config.enable_think:
                text += "<think>\n"  # force the LLM to think before answering
            else:
                text += "<answer>"  # force the LLM to answer

        # TODO: 应该没有必要，注意处理mask
        # TODO: special tokens add to config
        text = text.replace("<|im_end|>\n", "<|im_end|>")
        return [text], [messages]

    def _parse_response(self, response: str) -> List:
        pattern = (
            r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$"
            if self.pipeline_config.enable_think
            else r"^<answer>(.*?)</answer>$"
        )
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            think_content, action_content, actions = "INVALID", "INVALID", []  # 如何更好的处理invalid response?
            # print(f"Invalid response format: {response}")
            # yali: this may be cause potential crash
            # llm_response, actions = response, []
        else:
            if self.pipeline_config.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

            for special_token in self.pipeline_config.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()

            actions = [
                action.strip() for action in action_content.split(self.pipeline_config.action_sep) if action.strip()
            ]
            max_actions = 1

            if len(actions) > max_actions:
                actions = actions[:max_actions]  # Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.pipeline_config.action_sep + " ").join(actions)

        llm_response = (
            f"<think>\n{think_content}\n</think><answer>{action_content}</answer>"
            if self.pipeline_config.enable_think
            else f"<answer>{action_content}</answer>"
        )
        return llm_response, actions

    def start_input_queue_process(self):
        def process_input_queue(input_queue):
            while True:
                try:
                    command = input_queue.get_nowait()
                except Empty:
                    time.sleep(1)
                    continue
                if command == "stop":
                    self.logger.debug(f"{self.env_config['env_id']} stopped, episode_id: {self.episode_id}")
                    self.running = False
                    ray.get(
                        self.generate_scheduler.abort_request.remote(
                            DataProto(meta_info={"request_id": self.request_id})
                        )
                    )
                    self.request_id = None
                    break

        self.process_input_queue_thread = Thread(target=process_input_queue, args=(self.input_queue,))
        self.process_input_queue_thread.start()
