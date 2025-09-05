import matplotlib.pyplot as plt
import numpy as np
import os
import random
from roll.agentic.env.hanabi.config import HanabiConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv
from textwrap import dedent
import json
from typing import Optional, Dict, Any
import re
from PIL import Image
from collections import deque
import pyspiel
import warnings


class Hanabi(BaseDiscreteActionEnv):
    """Hanabi game environment using OpenSpiel."""

    def __init__(self, config: HanabiConfig = HanabiConfig()):
        self.config = config
        self.render_mode = config.render_mode
        self.built_in_opponent = config.built_in_opponent
        self.opponent_first_move = config.opponent_first_move
        self.include_opponent_turn = config.include_opponent_turn

        self.players = config.players
        self.colors = config.colors
        self.ranks = config.ranks
        self.hand_size = config.hand_size
        self.max_information_tokens = config.max_information_tokens
        self.max_life_tokens = config.max_life_tokens
        self.history_size = config.history_size

        if self.players != 2:
            raise ValueError(f"Current Hanabi only support 2 players, not {self.players} players.")

        self.game_parameters = {
            "players": self.players,
            "colors": self.colors,
            "ranks": self.ranks,
            "hand_size": self.hand_size,
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
        }

        BaseDiscreteActionEnv.__init__(self)

        self._env = pyspiel.load_game("hanabi", self.game_parameters)
        self.state = None
        self.num_steps = 0
        self.history = deque(maxlen=self.history_size)

    @property
    def current_player(self):
        if self.state is None:
            return 0
        return self.state.current_player()

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                self.game_parameters["seed"] = seed
                self._env = pyspiel.load_game("hanabi", self.game_parameters)
                self.state = self._env.new_initial_state()
                self.num_steps = 0
                self.history.clear()

                while self.state.is_chance_node():
                    outcomes_with_probs = self.state.chance_outcomes()
                    actions, probs = zip(*outcomes_with_probs)
                    action = random.choices(actions, weights=probs)[0]
                    self.state.apply_action(action)

                if self.built_in_opponent != "none" and self.opponent_first_move:
                    opponent_action = self._opponent_step()
                    self._step(opponent_action)

                return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        observation, rewards, done, info = self._step(action)
        # If chose to play with built-in opponent, we need to let the opponent take action
        if self.built_in_opponent != "none" and not done:
            opponent_action = self._opponent_step()
            observation, _rewards, done, info = self._step(opponent_action)
            rewards = [rewards[i] + _rewards[i] for i in range(2)]
        return observation, rewards, done, info

    def _step(self, action):
        if isinstance(action, str):
            action = self._string_to_action(action)
        if self.state is None or self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.history.append(f"player {self.current_player} select {self._action_to_string(self.current_player, action)}")
        self.state.apply_action(action)
        self.num_steps += 1
        chance_node_action = self._handle_chance_node()

        observation = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info(chance_node_action)
        if len(info) > 1:
            rewards = [0.1 if reward == 0 else reward for reward in rewards]
        return observation, rewards, done, info

    def _opponent_step(self):
        if self.built_in_opponent == "random":
            action = random.choice(list(self.get_all_actions().values()))
        else:
            raise ValueError(f"Invalid built-in teammate: {self.built_in_opponent}")
        print(f"Built-in {self.built_in_opponent} teammate taking action: {action}")
        return action

    def _handle_chance_node(self):
        if self.state.is_chance_node():
            outcomes_with_probs = self.state.chance_outcomes()
            actions, probs = zip(*outcomes_with_probs)
            action = random.choices(actions, weights=probs)[0]
            chance_node_action = self._action_to_string(self.current_player, action)
            self.state.apply_action(action)
            if self.state.is_chance_node():
                raise ValueError(f"the state is still a chance node\n{self.state}")
            return chance_node_action
        return None

    def get_prompt(self, mode="prefix", think=True, player_id=0):
        if mode == "prefix":
            prefix_prompt = self._get_prefix_prompt(think, player_id)
            return prefix_prompt
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True, player_id=0):
        system_prompt = "You are an AI agent that makes optimal decisions to achieve the highest score in the game of hanabi."
        
        color_set = ['Red(denoted by R)', 'Yellow(denoted by Y)', 'Green(denoted by G)', 'White(denoted by W)', 'Blue(denoted by B)']
        c, r = self.colors, self.ranks
        per_color = 3 + 2 * (r - 2) + 1
        total_cards = c * per_color
        color_names = ', '.join(color_set[:c])

        if r == 5:
            rank_range_str = f"Each color contains {per_color} cards: three of rank 1, two each of rank 2, rank 3, rank 4, and one of rank 5,"
        elif r == 4:
            rank_range_str = f"Each color contains {per_color} cards: three of rank 1, two each of rank 2, rank 3, and one of rank 4,"
        elif r == 3:
            rank_range_str = f"Each color contains {per_color} cards: three of rank 1, two of rank 2, and one of rank 3,"
        elif r == 2:
            rank_range_str = f"Each color contains {per_color} cards: three of rank 1, and one of rank 2,"
        else:
            raise ValueError(f"Invalid rank: {r}")

        rules = (
            f"1. Hanabi is a cooperative card game for 2 players, player 0 and player 1.\n"
            f"2. The deck consists of {c} colors: {color_names}, with ranks ranging from 1 to {r}. "
            f"{rank_range_str} for a total of {total_cards} cards.\n"
            f"3. Each player holds {self.hand_size} cards in hand. Players can observe the hand of the other player, but not their own.\n"
            f"4. There are {self.max_information_tokens} information tokens and {self.max_life_tokens} life tokens shared by both players.\n"
            f"5. The objective is to play cards in ascending order of rank, from 1 to {r}, to their corresponding color stacks, hence achieving the 'Fireworks'.\n"
            f"6. The players take turns to take one of the following actions:\n"
            "    a. <Play `i`>: play the i-th card from the player's own hand (0-indexed). If the card is sequential to the top card of its corresponding color stack, "
            "the move is valid and the card is added to the top of the stack, then both players receive 1 point. Otherwise, a life token is lost.\n"
            "    b. <Discard `i`>: discard the i-th card from the player's own hand and gain one information token.\n"
            "    c. <Reveal player +1 color `c`>: spend one information token to reveal all cards of color `c` in the other player's hand.\n"
            "    d. <Reveal player +1 rank `r`>: spend one information token to reveal all cards of rank `r` in the other player's hand.\n"
            f"7. After playing or discarding, the player receives a new card from the deck (if remaining).\n"
            f"8. The game ends when:\n"
            f"    a. If all color stacks are completed (i.e., all cards of rank {r} are played to their corresponding color stacks), "
            f"then both players finish the game with the highest possible total score of {r*c}.\n"
            f"    b. If deck is depleted, both players finish the game with a total score which equals the sum of the highest ranks of each color stack.\n"
            f"    c. If all life tokens are lost before the above two conditions are met, then both players lose all points they have earned so far, and finish the game with a total score of 0."
        )

        information = (
            f"1. You will be playing as the player {player_id}.\n"
            "2. In each of your turns, you will be provided with the current game state information, including the remaining life tokens and information tokens, "
            "the current color stacks, the remaining deck size, the discard pile, the hand of the other player, and the revealed information on your own hand.\n"
            "3. Known cards are denoted by their color and rank. For example, 'R2' means a red card of rank 2. "
            "4. The current color stacks are represented by the top card of each color stack. In particular, rank 0 denotes an empty stack. "
            "For example, 'Y0' means the yellow stack is still empty."
        )

        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = f"<answer><Discard Card 0></answer>"

        instructions = (
            f"Always choose only one action from the legal actions and output `{FORMAT_PROMPT}` with no extra text after you finish the thinking process. "
            f"For example, `{FORMAT_PROMPT_EXAMPLE}`. "
            "Strictly follow the above format. Responses that do not follow the format will result in immediate loss of all life tokens and end of the game."
        )

        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"RESPONSE INSTRUCTIONS:\n{instructions}\n\n"
        )
        prefix_prompt = {"system": system_prompt, "user": user_prompt}
        return prefix_prompt

    def get_all_actions(self):
        return self._get_legal_actions(self.current_player)

    def _get_legal_actions(self, player_id):
        legal_actions = dict()
        if self.state is None:
            return legal_actions
        actions = self.state.legal_actions(player_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(player_id, a)
        return legal_actions

    def _action_to_string(self, agent_id, action):
        action_str = self.state.action_to_string(agent_id, action)
        action_str = action_str.replace("(", "<").replace(")", ">")
        action_str = action_str.replace("Play", "Play card")
        action_str = action_str.replace("Discard", "Discard card")
        return action_str

    def _string_to_action(self, action_str):
        action_str = action_str.replace("<", "(").replace(">", ")")
        action_str = action_str.replace("Play card", "Play")
        action_str = action_str.replace("Discard card", "Discard")
        return self.state.string_to_action(action_str)

    def _get_info(self, chance_node_action):
        info = {}
        if self.state.is_terminal():
            returns = self.state.returns()
            info.update({
                "player_0_return": returns[0],
                "player_1_return": returns[1],
                "player_0_lose_for_wrong_format": 0,
                "player_1_lose_for_wrong_format": 0,
                "player_0_lose_for_overlong_response": 0,
                "player_1_lose_for_overlong_response": 0,
            })
            if chance_node_action:
                info["chance_node_action"] = chance_node_action
        else:
            if chance_node_action:
                info["chance_node_action"] = chance_node_action
        return info

    def get_losing_state(self, player_id: int=0, overlong_response: bool=False):
        observation = self.render()
        done = True
        returns = self.state.returns()
        if player_id == 0:
            reward = [-returns[0] - 10, 0]
            info = {
                "player_0_return": 0,
                "player_1_return": 0,
                "player_0_lose_for_wrong_format": 1,
                "player_1_lose_for_wrong_format": 0,
                "player_0_lose_for_overlong_response": 1 if overlong_response else 0,
                "player_1_lose_for_overlong_response": 0,
            }
        else:
            reward = [0, -returns[1] - 10]
            info = {
                "player_0_return": 0,
                "player_1_return": 0,
                "player_0_lose_for_wrong_format": 0,
                "player_1_lose_for_wrong_format": 1,
                "player_0_lose_for_overlong_response": 0,
                "player_1_lose_for_overlong_response": 1 if overlong_response else 0,
            }
        return observation, reward, done, info

    def render(self, mode: str = "text"):
        if mode == "text":
            return self._render_text()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _render_text(self):
        if self.state.is_terminal():
            return "The game is over."
        obs_dict = self._get_player_obs()
        discards = obs_dict['discards'] if obs_dict['discards'] else 'None'
        obs_str = [
            f"1. There are {obs_dict['life_tokens']} life tokens and {obs_dict['info_tokens']} information tokens remaining.",
            f"2. The top of the color stacks are: {obs_dict['fireworks']}.",
            f"3. {obs_dict['deck_size']} cards remain in the draw pile.",
            f"4. The discard pile currently contains: {discards}.",
        ]

        obs_str.append("5. The other player's hand:")
        for i, card_info in obs_dict['other_info'].items():
            colors = ', '.join(card_info['colors'])
            ranks = ', '.join(card_info['digits'])
            visible_card = card_info['visible_card']
            obs_str.append(
                f"    - Card {i} ({visible_card}): the other player believes it is one of the colors [{colors}] and one of the ranks [{ranks}]."
            )

        obs_str.append("6. Your own hand, based on the revealed information:")
        for i, card_info in obs_dict['current_player_info'].items():
            colors = ', '.join(card_info['colors'])
            ranks = ', '.join(card_info['digits'])
            obs_str.append(f"    - Card {i}: one of the colors [{colors}] and one of the ranks [{ranks}].")
        return "\n".join(obs_str)

    def _render_rgb_array(self):
        warnings.warn("Hanabi does not support image rendering yet.")
        return None

    def _get_player_obs(self):
        obs = self.state.observation_string()
        lines = obs.strip().split('\n')
        result = {
            "life_tokens": None,
            "info_tokens": None,
            "fireworks": "",
            "deck_size": None,
            "discards": "",
            "current_player_hand": [],
            "other_hands": [],
            "current_player_info": {},
            "other_info": {},
        }
        in_hands = False
        hands_section = []
        current_player_index = None
        index = None

        for line in lines:
            if line.startswith("Life tokens:"):
                result["life_tokens"] = int(line.split(":")[1].strip())
            elif line.startswith("Info tokens:"):
                result["info_tokens"] = int(line.split(":")[1].strip())
            elif line.startswith("Fireworks:"):
                result["fireworks"] = line.split(":", 1)[1].strip()
            elif line.strip() == "Hands:":
                in_hands = True
            elif line.strip() == "Cur player":
                current_player_index = len(hands_section)
            elif line.startswith("Deck size:"):
                result["deck_size"] = int(line.split(":")[1].strip())
                in_hands = False
            elif line.startswith("Discards:"):
                result["discards"] = line.split(":", 1)[1].strip()
            elif in_hands:
                if line.strip() != "-----":
                    hands_section.append(line.strip())
                else:
                    index = len(hands_section)

        if current_player_index == 0:
            result["other_hands"] = hands_section[index:]
            result["current_player_hand"] = hands_section[:index]
        else:
            result["other_hands"] = hands_section[:index]
            result["current_player_hand"] = hands_section[index:]

        for i, card in enumerate(result['current_player_hand']):
            card_info = card.split("|")[-1]
            letters = re.findall(r"[A-Za-z]", card_info)
            digits = re.findall(r"\d", card_info)
            result['current_player_info'][f'{i}'] = {'digits': digits, 'colors': letters}

        for i, card in enumerate(result['other_hands']):
            visible_card = card.split('||')[0].strip()
            card_info = card.split("|")[-1]
            letters = re.findall(r"[A-Za-z]", card_info)
            digits = re.findall(r"\d", card_info)
            result['other_info'][f'{i}'] = {'visible_card': visible_card, 'digits': digits, 'colors': letters}

        return result

    def close(self):
        """Close the environment."""
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()


if __name__ == "__main__":
    # Basic unit test
    print("-" * 100)
    print("Basic unit test:")
    print("-" * 100)
    env = Hanabi()
    
    results = []
    for i in range(1):
        print('-' * 100)
        print(f'Episode {i}')
        print('-' * 100)
        observation = env.reset()
        done = False
        while not done:
            prefix_prompt = env.get_prompt(mode="prefix")
            print(f"System prompt: \n{prefix_prompt['system']}")
            print(f"User prompt: \n{prefix_prompt['user']}")
            print(f"observation: \n{observation}")
            action = random.choice(list(env.get_all_actions().values()))
            print(f"Player {env.current_player} legal actions: {env.get_all_actions()}")
            print(f"Player {env.current_player} taking action: {action}")
            observation, rewards, done, info = env.step(action)
            print(f"rewards: {rewards}")
            print(f"done: {done}")
            print(f"info: {info}")
            print("-" * 100)
        results.append(info.get('returns', [0, 0]))
    print("Average returns: ", [sum(r[i] for r in results) / len(results) for i in range(2)])
    print("-" * 100)