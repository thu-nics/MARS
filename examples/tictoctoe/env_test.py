import sys

sys.path.insert(0, "/Users/zheyue/Workspace/Projects/Post-Training/ROLL/")


rollout_cache = {
    "env_id": 31,
    "history": [
        {
            "state": "___\n___\n___",
            "actions_left": 5,
            "legal_actions": {
                0: "<X(0,0)>",
                1: "<X(0,1)>",
                2: "<X(0,2)>",
                3: "<X(1,0)>",
                4: "<X(1,1)>",
                5: "<X(1,2)>",
                6: "<X(2,0)>",
                7: "<X(2,1)>",
                8: "<X(2,2)>",
            },
            "actions": ["<X(1,1)>"],
            "reward": -0.05,
            "info": {},
            "llm_response": "<think>Okay, so I need to play tic-tac-toe as X. The game starts with all cells empty. The first move is mine. Let me think about the best strategy here. Since it's the opening move, I should aim to take the center, because it's the most strategic position. It gives the most lines to win. Wait, but maybe the center is not the first move? Or is it? Wait, in tic-tac-toe, the center is a key position. Let me confirm. If I take the center first, that's a good idea. So the center is (1,1). So the action would be X(1,1). That makes sense. Alternatively, maybe taking a corner is better? But I think center is better for controlling the game. So I'll go with center.</think><answer><X(1,1)></answer>",
            "llm_raw_response": "<think>\nOkay, so I need to play tic-tac-toe as X. The game starts with all cells empty. The first move is mine. Let me think about the best strategy here. Since it's the opening move, I should aim to take the center, because it's the most strategic position. It gives the most lines to win. Wait, but maybe the center is not the first move? Or is it? Wait, in tic-tac-toe, the center is a key position. Let me confirm. If I take the center first, that's a good idea. So the center is (1,1). So the action would be X(1,1). That makes sense. Alternatively, maybe taking a corner is better? But I think center is better for controlling the game. So I'll go with center.\n</think>\n\n<answer><X(1,1)></answer>",
        },
        {
            "state": "___\nOX_\n___",
            "actions_left": 4,
            "legal_actions": {
                0: "<X(0,0)>",
                1: "<X(0,1)>",
                2: "<X(0,2)>",
                5: "<X(1,2)>",
                6: "<X(2,0)>",
                7: "<X(2,1)>",
                8: "<X(2,2)>",
            },
            "actions": ["<X(0,0)>"],
            "reward": -0.05,
            "info": {},
            "llm_response": "<think>Okay, let's see. The game is tic-tac-toe, and I'm playing as X. The previous move was the user placing an O in (1,1). Now the current board is:\n\n___\nOX_\n___\n\nSo the first row is all empty, the second row has O in column 1, and the third row is empty. My turn to place an X.\n\nFirst, I need to check if there's a winning move. Let's look for possible lines where placing an X could complete three in a row, column, or diagonal.\n\nLooking at the second row: there's an O in (1,1). If I place an X in (1,0) or (1,2), that row would have two Xs and one O, but not a win. Wait, no, the second row is O X _ so if I put X in (1,2), then the row would be O X X, which is two in a row but not three. So that's not a win.\n\nWhat about columns? Let's check column 1. The second row has an O there. The first row's column 1 is empty, third row's column 1 is empty. So if I place X in (0,1) or (2,1), that column would have one X. Not enough.\n\nLooking for diagonals. The main diagonal is (0,0), (1,1), (2,2). The center is O, so if I can get two Xs there. But currently, (0,0) and (2,2) are empty. If I place X in (0,0) and (2,2), but that's two moves. But I can only place one now. So not a win here.\n\nAnother diagonal is (0,2), (1,1), (2,0). The center is O, so same issue.\n\nMaybe check rows. The first row is all empty. If I place X in (0,0), (0,1), or (0,2), but that's just starting. The third row is empty. Maybe if I can block the opponent's possible win.\n\nWait, the opponent's last move was O at (1,1). They might be trying to create a line. Let me think: if they place O in (1,1), maybe they are trying to complete a diagonal or something. But with only one O, they can't win yet.\n\nSo, what's the best move for me? Let's consider possible moves. The legal actions are X(0,0), X(0,1), X(0,2), X(1,2), X(2,0), X(2,1), X(2,2).\n\nLooking for the center. Wait, the center is (1,1), which is already O. So I can't place there. So possible moves are the corners and (1,2).\n\nIf I place X in (0,0), that's a corner. Then maybe I can create a line. Alternatively, (0,2) or (2,0) or (2,2). Also, (1,2) is the top of the second row.\n\nAlternatively, maybe the best is to take the center, but it's already taken. So maybe the best is to take a corner. Let's think: if I take (0,0), then the first row has X at (0,0). Then, if the opponent plays in (0,1) or (0,2), I can complete the row. But maybe that's too far.\n\nAlternatively, maybe the best is to take (1,2) to block the opponent. Wait, the opponent has O at (1,1). If they place O in (1,2), that would be two in a row. But they can't, because I'm the one placing X. Wait, the opponent is O, and they just placed at (1,1). So maybe they are trying to create a line. But with only one O, they can't.\n\nAlternatively, maybe the best is to take the center. But it's already O. So maybe the best is to take a corner. Let's pick (0,0). That's a good move to start. Or maybe (2,2) for the other diagonal.\n\nAlternatively, maybe (1,2) is a good move to block. Let me think: if I place X at (1,2), then the second row would be O X X. That's two in a row. But that's not a win. But maybe that's a way to start building a line.\n\nAlternatively, maybe the best is to take the center. But it's already taken. So maybe take a corner. Let me pick (0,0). So the answer is X(0,0).</think><answer><X(0,0)></answer>",
            "llm_raw_response": "<think>\nOkay, let's see. The game is tic-tac-toe, and I'm playing as X. The previous move was the user placing an O in (1,1). Now the current board is:\n\n___\nOX_\n___\n\nSo the first row is all empty, the second row has O in column 1, and the third row is empty. My turn to place an X.\n\nFirst, I need to check if there's a winning move. Let's look for possible lines where placing an X could complete three in a row, column, or diagonal.\n\nLooking at the second row: there's an O in (1,1). If I place an X in (1,0) or (1,2), that row would have two Xs and one O, but not a win. Wait, no, the second row is O X _ so if I put X in (1,2), then the row would be O X X, which is two in a row but not three. So that's not a win.\n\nWhat about columns? Let's check column 1. The second row has an O there. The first row's column 1 is empty, third row's column 1 is empty. So if I place X in (0,1) or (2,1), that column would have one X. Not enough.\n\nLooking for diagonals. The main diagonal is (0,0), (1,1), (2,2). The center is O, so if I can get two Xs there. But currently, (0,0) and (2,2) are empty. If I place X in (0,0) and (2,2), but that's two moves. But I can only place one now. So not a win here.\n\nAnother diagonal is (0,2), (1,1), (2,0). The center is O, so same issue.\n\nMaybe check rows. The first row is all empty. If I place X in (0,0), (0,1), or (0,2), but that's just starting. The third row is empty. Maybe if I can block the opponent's possible win.\n\nWait, the opponent's last move was O at (1,1). They might be trying to create a line. Let me think: if they place O in (1,1), maybe they are trying to complete a diagonal or something. But with only one O, they can't win yet.\n\nSo, what's the best move for me? Let's consider possible moves. The legal actions are X(0,0), X(0,1), X(0,2), X(1,2), X(2,0), X(2,1), X(2,2).\n\nLooking for the center. Wait, the center is (1,1), which is already O. So I can't place there. So possible moves are the corners and (1,2).\n\nIf I place X in (0,0), that's a corner. Then maybe I can create a line. Alternatively, (0,2) or (2,0) or (2,2). Also, (1,2) is the top of the second row.\n\nAlternatively, maybe the best is to take the center, but it's already taken. So maybe the best is to take a corner. Let's think: if I take (0,0), then the first row has X at (0,0). Then, if the opponent plays in (0,1) or (0,2), I can complete the row. But maybe that's too far.\n\nAlternatively, maybe the best is to take (1,2) to block the opponent. Wait, the opponent has O at (1,1). If they place O in (1,2), that would be two in a row. But they can't, because I'm the one placing X. Wait, the opponent is O, and they just placed at (1,1). So maybe they are trying to create a line. But with only one O, they can't.\n\nAlternatively, maybe the best is to take the center. But it's already O. So maybe the best is to take a corner. Let's pick (0,0). That's a good move to start. Or maybe (2,2) for the other diagonal.\n\nAlternatively, maybe (1,2) is a good move to block. Let me think: if I place X at (1,2), then the second row would be O X X. That's two in a row. But that's not a win. But maybe that's a way to start building a line.\n\nAlternatively, maybe the best is to take the center. But it's already taken. So maybe take a corner. Let me pick (0,0). So the answer is X(0,0).\n</think>\n\n<answer><X(0,0)></answer>",
        },
        {
            "state": "X__\nOX_\n_O_",
            "actions_left": 3,
            "legal_actions": {1: "<X(0,1)>", 2: "<X(0,2)>", 5: "<X(1,2)>", 6: "<X(2,0)>", 8: "<X(2,2)>"},
            "actions": ["<X(2,2)>"],
            "reward": 0.95,
            "info": {"player_0_return": 1.0, "player_1_return": -1.0, "winner": 0, "lose_for_wrong_format": 0},
            "llm_response": "<think>Okay, let's see. The game is tic-tac-toe, and I'm playing as X. The current board state is:\n\nX__\nOX_\n_O_\n\nSo, the first row is X, then two blanks. The second row is O, X, _. The third row is _, O, _.\n\nMy turn. I need to choose a move. Let me think about possible winning moves or blocking the opponent.\n\nFirst, check if I can win immediately. Let's see where X's are. I have X in (0,0), and (1,1) was previously filled by O, but wait, the current state shows the second row as O X _, so O is at (1,0), X at (1,1)? Wait, no. Wait the second row is OX_, so that's O at (1,0), X at (1,1), and blank at (1,2). Wait, but in the previous turn, the user placed X at (0,0). Then the current state is:\n\nRow 0: X _ _\nRow 1: O X _\nRow 2: _ O _\n\nSo, looking for possible lines. For me (X), are there any two X's in a row, column, or diagonal that can be completed with my next move?\n\nLooking at row 0: X is at (0,0), others are empty. So if I place X at (0,1) or (0,2), that's a possible line. But not yet a win.\n\nRow 1: O, X, _. So if I put X at (1,2), that would make two X's in row 1. But not three. Wait, no. Row 1 has O, X, and then if I put X at (1,2), that would be O, X, X. But that's not three in a row. Wait, no, the row is O, X, _, so if I put X at (1,2), then row 1 would be O, X, X. That's not a win. But maybe a diagonal?\n\nLooking at diagonals. The main diagonal is (0,0), (1,1), (2,2). Currently, (0,0) is X, (1,1) is X (from the second row's X at (1,1)? Wait, no. Wait the second row is O X _, so (1,0) is O, (1,1) is X, (1,2) is blank. So the main diagonal is (0,0)=X, (1,1)=X, (2,2)=_. If I place X at (2,2), that would complete the diagonal. So that's a winning move. So that's a possibility.\n\nAlternatively, the other diagonal is (0,2), (1,1), (2,0). Currently, (0,2) is blank, (1,1) is X, (2,0) is blank. So if I can get two in a row there. But I need to check if there's a way to complete that.\n\nBut the main diagonal is (0,0), (1,1), (2,2). So if I put X at (2,2), that would make three in a row. So that's a winning move. So that's a good option.\n\nAlternatively, looking for possible threats from O. O has O at (1,0) and (2,1). Are they about to make a line? Let's see. For O, if they have two in a row, column, or diagonal, then they need to block.\n\nLooking at O's positions: (1,0) and (2,1). Are there any lines they can complete? For example, if O places at (0,0), but that's already X. Or maybe (1,0) and (2,1) are part of a diagonal? Let's see. The diagonal from (1,0) to (2,1) would be part of the diagonal (0,0), (1,1), (2,2), but O is at (1,0) and (2,1). Wait, that's not a diagonal. Wait, (1,0) is in row 1, column 0. (2,1) is row 2, column 1. So that's not a diagonal. Maybe a column? Column 0 has X at (0,0), O at (1,0), and blank at (2,0). So O has one in column 0. Not a threat.\n\nAlternatively, O has (1,0) and (2,1). Maybe if they place at (0,0), but that's already X. So maybe O isn't in a position to win immediately. So my main goal is to either win or block O.\n\nSo possible moves for me: (2,2) to complete the diagonal, or (0,1) to start a row, or (0,2) to start a row, or (1,2) to complete row 1, or (2,0) to start column 0.\n\nBut the available moves are X(0,1), X(0,2), X(1,2), X(2,0), X(2,2). So the options are those.\n\nIf I choose (2,2), that would be a winning move. So that's the best option. So I should go for that.\n\nAlternatively, if I don't have a winning move, I should block O's possible moves. Let's check if O can win in the next move. For example, if O is about to make a line.\n\nLooking at O's current positions: (1,0) and (2,1). If O places at (0,0), that would be a diagonal, but (0,0) is X. So no. Or O could place at (1,2) to make a column? Wait, O is at (1,0) and (2,1). If O places at (0,1), that would be a column? No. Wait, column 1 has (0,1) blank, (1,1) is X, (2,1) is O. So O has one in column 1. Not a threat.\n\nAlternatively, O could complete a row. For example, row 2 has (2,0) blank, (2,1) O, (2,2) blank. So if O places at (2,0) or (2,2), but that's not possible. So O isn't in a position to win.\n\nTherefore, the best move is to take (2,2) to complete the diagonal and win. So the answer should be X(2,2).</think><answer><X(2,2)></answer>",
            "llm_raw_response": "<think>\nOkay, let's see. The game is tic-tac-toe, and I'm playing as X. The current board state is:\n\nX__\nOX_\n_O_\n\nSo, the first row is X, then two blanks. The second row is O, X, _. The third row is _, O, _.\n\nMy turn. I need to choose a move. Let me think about possible winning moves or blocking the opponent.\n\nFirst, check if I can win immediately. Let's see where X's are. I have X in (0,0), and (1,1) was previously filled by O, but wait, the current state shows the second row as O X _, so O is at (1,0), X at (1,1)? Wait, no. Wait the second row is OX_, so that's O at (1,0), X at (1,1), and blank at (1,2). Wait, but in the previous turn, the user placed X at (0,0). Then the current state is:\n\nRow 0: X _ _\nRow 1: O X _\nRow 2: _ O _\n\nSo, looking for possible lines. For me (X), are there any two X's in a row, column, or diagonal that can be completed with my next move?\n\nLooking at row 0: X is at (0,0), others are empty. So if I place X at (0,1) or (0,2), that's a possible line. But not yet a win.\n\nRow 1: O, X, _. So if I put X at (1,2), that would make two X's in row 1. But not three. Wait, no. Row 1 has O, X, and then if I put X at (1,2), that would be O, X, X. But that's not three in a row. Wait, no, the row is O, X, _, so if I put X at (1,2), then row 1 would be O, X, X. That's not a win. But maybe a diagonal?\n\nLooking at diagonals. The main diagonal is (0,0), (1,1), (2,2). Currently, (0,0) is X, (1,1) is X (from the second row's X at (1,1)? Wait, no. Wait the second row is O X _, so (1,0) is O, (1,1) is X, (1,2) is blank. So the main diagonal is (0,0)=X, (1,1)=X, (2,2)=_. If I place X at (2,2), that would complete the diagonal. So that's a winning move. So that's a possibility.\n\nAlternatively, the other diagonal is (0,2), (1,1), (2,0). Currently, (0,2) is blank, (1,1) is X, (2,0) is blank. So if I can get two in a row there. But I need to check if there's a way to complete that.\n\nBut the main diagonal is (0,0), (1,1), (2,2). So if I put X at (2,2), that would make three in a row. So that's a winning move. So that's a good option.\n\nAlternatively, looking for possible threats from O. O has O at (1,0) and (2,1). Are they about to make a line? Let's see. For O, if they have two in a row, column, or diagonal, then they need to block.\n\nLooking at O's positions: (1,0) and (2,1). Are there any lines they can complete? For example, if O places at (0,0), but that's already X. Or maybe (1,0) and (2,1) are part of a diagonal? Let's see. The diagonal from (1,0) to (2,1) would be part of the diagonal (0,0), (1,1), (2,2), but O is at (1,0) and (2,1). Wait, that's not a diagonal. Wait, (1,0) is in row 1, column 0. (2,1) is row 2, column 1. So that's not a diagonal. Maybe a column? Column 0 has X at (0,0), O at (1,0), and blank at (2,0). So O has one in column 0. Not a threat.\n\nAlternatively, O has (1,0) and (2,1). Maybe if they place at (0,0), but that's already X. So maybe O isn't in a position to win immediately. So my main goal is to either win or block O.\n\nSo possible moves for me: (2,2) to complete the diagonal, or (0,1) to start a row, or (0,2) to start a row, or (1,2) to complete row 1, or (2,0) to start column 0.\n\nBut the available moves are X(0,1), X(0,2), X(1,2), X(2,0), X(2,2). So the options are those.\n\nIf I choose (2,2), that would be a winning move. So that's the best option. So I should go for that.\n\nAlternatively, if I don't have a winning move, I should block O's possible moves. Let's check if O can win in the next move. For example, if O is about to make a line.\n\nLooking at O's current positions: (1,0) and (2,1). If O places at (0,0), that would be a diagonal, but (0,0) is X. So no. Or O could place at (1,2) to make a column? Wait, O is at (1,0) and (2,1). If O places at (0,1), that would be a column? No. Wait, column 1 has (0,1) blank, (1,1) is X, (2,1) is O. So O has one in column 1. Not a threat.\n\nAlternatively, O could complete a row. For example, row 2 has (2,0) blank, (2,1) O, (2,2) blank. So if O places at (2,0) or (2,2), but that's not possible. So O isn't in a position to win.\n\nTherefore, the best move is to take (2,2) to complete the diagonal and win. So the answer should be X(2,2).\n</think>\n\n<answer><X(2,2)></answer>",
        },
        {"state": "X__\nOX_\n_OX", "actions_left": 2, "legal_actions": {}},
    ],
    "group_id": 31,
    "tag": "TicTacToe",
    "penalty": 0,
    "frames": [],
}


import copy
from dataclasses import dataclass
from dataclasses import field
from itertools import zip_longest
from typing import Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer

from roll.agentic.env import REGISTERED_ENV_CONFIGS
from roll.agentic.env import REGISTERED_ENVS
from roll.distributed.scheduler.protocol import DataProto


def pad_to_length(tensor: torch.Tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(0, length)
        return tensor[indices]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


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


class RolloutTest:
    def __init__(self, env_config, pipeline_config, tokenizer, processor=None):

        self.env_config = env_config
        self.env_entry = copy.deepcopy(self.env_config)
        self.env_entry["env"] = REGISTERED_ENVS[self.env_entry["env_class"]](self.env_entry["config"])
        self.env_entry["status"] = EnvStatus()

        self.pipeline_config = pipeline_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.rollout_cache = rollout_cache

    def _format_messages(self, env_output: Dict, prepare_for_update: bool, use_raw_llm_response: bool):
        if "state" in env_output["history"][-1] and (not use_raw_llm_response and prepare_for_update):
            # when prepare for update, we do not add the state from the n+1 turn to the trajectory
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
            turn_idx_content = f"\n\nTurn {idx + 1}:\n\n"
            messages[-1]["content"] += turn_idx_content
            if "state" in content:
                messages[-1]["content"] += (
                    f"GAME STATE:\n{content['state']}\n\n"
                    f"LEGAL ACTIONS:\n{', '.join(content['legal_actions'].values())}.\n\n"
                )
            if "llm_raw_response" in content:
                #       改成actions合理吗？
                messages.append(
                    {
                        "role": "assistant",
                        "content": content["llm_response"] if not use_raw_llm_response else content["llm_raw_response"],
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
                text += "<think>"  # force the LLM to think before answering
            else:
                text += "<answer>"  # force the LLM to answer

        # TODO: 应该没有必要，注意处理mask
        # TODO: special tokens add to config
        text = text.replace("<|im_end|>\n", "<|im_end|>")
        return [text], [messages]

    def formulate_rollouts(self):
        """
        1. 每个env的trajectory 应该是一个rollout
        2. 每个rollout 应该是一个List[Dict]
        3. 每个Dict 应该是一个step的信息
        """
        print("rollout cache: ", self.rollout_cache)

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
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=True
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
                    continue
                if k not in custom_metric:
                    custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            # env_metric[k] = np.sum(v) / len(self.rollout_cache['history'])
            env_metric[k] = np.sum(v)

        self.rollout_cache["history"][-1]["metrics"] = custom_metric
        env_metric = {f"env/{entry['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        self.rollout_cache["metrics"] = env_metric
        llm_inputs.meta_info = {"metrics": env_metric}
        return llm_inputs


from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf


from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig


def main(config_path, config_name):

    initialize(config_path=config_path, job_name="app")
    cfg = compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))
    ppo_config.__post_init__()

    model_name_or_path = "Qwen/Qwen3-4B"
    env_config = ppo_config.train_env_manager.env_configs[0][0]
    tester = RolloutTest(
        env_config=env_config,
        pipeline_config=ppo_config,
        tokenizer=AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            split_special_tokens=False,
            trust_remote_code=True,
            padding_side="left",
        ),
    )
    llm_inputs = tester.formulate_rollouts()
    print("llm_inputs: ", llm_inputs)


if __name__ == "__main__":
    main(config_path="./", config_name="agentic_val_tictactoe_tune")
