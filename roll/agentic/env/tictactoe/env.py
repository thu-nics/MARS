import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel
import random
from roll.agentic.env.tictactoe.config import TicTacToeConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv
from textwrap import dedent
from openai import OpenAI
import json
from typing import Optional, Dict, Any
import re

class TicTacToe(BaseDiscreteActionEnv):
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(self, config: TicTacToeConfig = TicTacToeConfig()):
        # Using mappings directly from config
        self.config = config
        self.render_mode = config.render_mode
        self.random_opponent = config.random_opponent

        BaseDiscreteActionEnv.__init__(self)
        
        self._env = pyspiel.load_game("tic_tac_toe")
        self.state = None

    @property
    def current_player(self):
        if self.state is None:
            return 0
        return self.state.current_player()

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                self.state = self._env.new_initial_state()
                return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action_str):
        observations, _, done, info = self._step(action_str)
        # If the opponent is random, we need to let the opponent take action
        if self.current_player == 1 and self.random_opponent and not done:
            observations, _, done, info = self._step(random.choice(list(self.get_all_actions().values())))
        reward = 0
        if done:
            reward = self.state.returns()[0]
        return observations, reward, done, info

    def _step(self, action_str):
        action = self._string_to_action(action_str)
        if self.state is None or self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.state.apply_action(action)
        observations = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info()
        return observations, rewards, done, info

    def get_prompt(self, mode="prefix", think=True):
        if mode == "prefix":
            prefix_prompt = self._get_prefix_prompt(think)
            return prefix_prompt
        elif mode == "state":
            state_prompt = self._get_state_prompt()
            return state_prompt
        elif mode == "action":
            action_prompt = self._get_action_prompt()
            return action_prompt
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True):
        system_prompt = "You are an AI agent that makes optimal decisions to win in the game of tic-tac-toe."
        rules = (
                "1. Tic-tac-toe is a two-player board game played on a three-by-three grid. "
                "The grid is 0-indexed, where (0,0) is the top-left corner and (2,2) is the bottom-right corner.\n"
                "2. Two players take turns placing their marks X and O in empty cells of the grid.\n"
                "3. The player who first places three of their marks in a horizontal, vertical, or diagonal line wins.\n"
                "4. If all cells are filled and no player wins, the game ends in a draw."
                )
        # mark = "X" if self.current_player == 0 else "O"
        mark = "X"
        opponent_mark = "O" if mark == "X" else "X"
        information = (
            f"1. Your mark is {mark}. You are competing with another player controlling the mark {opponent_mark}.\n"
            "2. In each of your turns:\n"
            "   a. The game state demonstrates the current board with a three-line text grid, where 'X' and 'O' are the marks of the two players, and '_' represents empty cells.\n"
            "   b. You need to chose an action to place your mark in an empty cell, based on the given game state and the history of your decisions.\n"
            f"   c. All legal actions are provided in the format of `<{mark}({{row}},{{column}})>`, where `{mark}` is your mark, "
            "and {row} and {column} are integers indicating the row and column of the cell to place your mark."
        )
        # if think:
        #     FORMAT_PROMPT = "<think>[your thinking]</think><answer>[your action]</answer>"
        #     FORMAT_PROMPT_EXAMPLE = f"<think>Okay, let's see. I'm playing a game of tic-tac-toe. This is my first turn. I will take <X(0,0)> because ...</think><answer><X(0,0)></answer>"
        # else:
        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = f"<answer><X(0,0)></answer>"
        instructions = (
            f"Always choose only one action from the legal actions and output {FORMAT_PROMPT} with no extra text. "
            f"For example, `{FORMAT_PROMPT_EXAMPLE}`. "
            "Strictly follow this format. Responses that do not follow the format will result in immediate loss of the game.\n"
            "You don't have too much time to think. Think shortly. Do not overthink."
        )
        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\n{information}\n\n"
            f"RESPONSE INSTRUCTIONS:\n{instructions}\n\n"
        )
        prefix_prompt = {
            "system": system_prompt,
            "user": user_prompt
        }
        return prefix_prompt

    def _get_state_prompt(self):
        state_prompt = (
            "This is the current snapshot of the board, "
            "where 'X' and 'O' are the marks of the two players, and '_' represents empty cells."
        )
        return state_prompt

    def _get_action_prompt(self):
        action_prompt = (
            "Each action is represented as <{mark}({row},{column})>, where {mark} is your mark, "
            "and {row} and {column} are integers indicating the row and column of the cell to place your mark."
        )
        return action_prompt

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

    def _action_to_string(self, player_id, action):
        mark = "X" if player_id == 0 else "O"
        row = action // 3
        column = action % 3
        return f"<{mark}({row},{column})>"

    def _string_to_action(self, action_str):
        mark = action_str[1]
        row = int(action_str[3])
        column = int(action_str[5])
        return row * 3 + column

    def _get_info(self):
        if self.state is None or not self.state.is_terminal():
            return {}
        returns = self.state.returns()
        winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
        return {
            "player_0_return": returns[0],
            "player_1_return": returns[1],
            "winner": winner,
            "lose_for_wrong_format": 0
        }

    def get_losing_state(self):
        observation = self.render()
        reward = -1
        done = True
        info = {
            "player_0_return": -1,
            "player_1_return": 1,
            "winner": 1,
            "lose_for_wrong_format": 1
        }
        return observation, reward, done, info

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            return self._render_text()
        elif render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self):
        """Render the game state as text."""
        if self.state is None:
            return "___\n___\n___"
        board = np.array([list(line) for line in str(self.state).strip().split("\n")])
        text_repr = []
        for i in range(3):
            row = []
            for j in range(3):
                piece = board[i][j]
                if piece == '.':
                    row.append('_')
                elif piece == 'x':
                    row.append('X')
                elif piece == 'o':
                    row.append('O')
                else:
                    row.append(piece)
            text_repr.append(''.join(row))
        return '\n'.join(text_repr)

    def _render_rgb_array(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.invert_yaxis()
        for x in range(1, 3):
            ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color='black', linewidth=2)
        for y in range(1, 3):
            ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color='black', linewidth=2)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'])
        ax.set_yticklabels(['0', '1', '2'])
        if self.state is not None:
            board = np.array([list(line) for line in str(self.state).strip().split("\n")])
            for i in range(3):
                for j in range(3):
                    piece = board[i][j]
                    if piece != '.':
                        color = 'red' if piece == 'x' else 'blue'
                        ax.text(j, i, piece.upper(), fontsize=30, ha='center', va='center', color=color)
        ax.set_aspect('equal')
        return fig

    def close(self):
        """Close the environment."""
        if hasattr(self, '_env') and self._env is not None:
            self._env.close()

if __name__ == "__main__":
    # Basic unit test
    print("-"*100)
    print("Basic unit test:")
    print("-"*100)
    env = TicTacToe()
    env.reset()
    done = False
    while not done:
        print("-"*100)
        prefix_prompt = env.get_prompt(mode='prefix')
        print(f"System prompt: \n{prefix_prompt['system']}")
        print(f"User prompt: \n{prefix_prompt['user']}")
        print(f"State prompt: \n{env.get_prompt(mode='state')}")
        print("-"*100)
        action = random.choice(list(env.get_all_actions().values()))
        # action = "X(0,0)"
        print(f"Player {env.current_player} taking action: {action}")
        observations, rewards, done, info = env.step(action)
        print(f"observations: \n{observations}")
        print(f"rewards: {rewards}")
        print(f"done: {done}")
        print(f"info: {info}")
        print("-"*100)

    # Test with a simple agent
    print("-"*100)
    print("Test with a simple agent:")
    print("-"*100)
    # Initialize OpenAI client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Call LLM to get action based on instruction
    def get_action_from_llm(prompt):
        response = client.chat.completions.create(
            model="/mnt/h_public/algm/models/Qwen3-4B",
            messages=prompt
        )
        raw_content = response.choices[0].message.content
        if raw_content is None:
            raw_content = ""

        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, raw_content, re.DOTALL)
        action_content = match.group(1) if match else "INVALID"
        try:
            action_json = json.loads(action_content)
            return action_json["action"], raw_content
        except Exception as e:
            print("Failed to parse LLM response:", raw_content)
            print("Extracted content:", action_content)
            raise e

    env = TicTacToe()
    num_steps = 5  # Number of steps to simulate
    from tqdm import tqdm
    env.reset()
    for step in tqdm(range(num_steps)):
        prompt = env.get_prompt(mode="prefix")
        prompt = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]
        print(f"========== Step {step + 1} ==========")
        print(f"System prompt: \n{prompt[0]['content']}")
        print(f"User prompt: \n{prompt[1]['content']}\n")
        action, raw_content = get_action_from_llm(prompt)
        print(f"LLM Response: \n{raw_content}")
        print(f"LLM Action: {action}")
        observations, rewards, done, info = env.step(action)
        print(f"observations: \n{observations}")
        print(f"rewards: {rewards}")
        print(f"done: {done}")
        print(f"info: {info}")

        if done:
            break
