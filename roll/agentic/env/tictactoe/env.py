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


class TicTacToe(BaseDiscreteActionEnv):
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(self, config: TicTacToeConfig = TicTacToeConfig()):
        # Using mappings directly from config
        self.config = config
        self.ACTION_LOOKUP = config.action_lookup
        self.render_mode = config.render_mode

        BaseDiscreteActionEnv.__init__(self)
        
        self._env = pyspiel.load_game("tic_tac_toe")
        self.state = None

    @property
    def current_player(self):
        return self.state.current_player()

    def reset(self, seed=0):
        try:
            with all_seed(seed):
                self.state = self._env.new_initial_state()
                return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def step(self, action_str):
        action = self._string_to_action(action_str)
        if self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.state.apply_action(action)
        observations = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info()
        return observations, rewards, done, info

    def render(self, mode=None):
        render_mode = mode if mode is not None else self.render_mode
        if render_mode == "text":
            return self._render_text()
        elif render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Invalid mode: {render_mode}")

    def get_prompt(self):
        system_prompt = "You are an AI agent that makes optimal decisions in the game of tic-tac-toe."
        rules = (
                "1. Tic-tac-toe is a two-player board game played on a three-by-three grid. "
                "The grid is 0-indexed, where (0,0) is the top-left corner and (2,2) is the bottom-right corner.\n"
                "2. Two players take turns placing their marks X and O in empty cells of the grid.\n"
                "3. The player who first places three of their marks in a horizontal, vertical, or diagonal line wins.\n"
                "4. If all cells are filled and no player wins, the game ends in a draw."
                )
        mark = "X" if self.current_player == 0 else "O"
        all_actions = ", ".join(list(self.get_all_actions().values()))
        instructions = dedent(
            f"""\
            Now it is your turn to choose an action. You should output your action in the following JSON format:
            ```json
            {{
                "action": "{mark}(i,j)"
            }}
            ```
            where i is the row index and j is the column index."""
        )
        user_prompt = (
            f"GAME RULES:\n{rules}\n\n"
            f"PLAYER INFORMATION:\nYour mark is {mark}.\n\n"
            f"GAME STATE:\n{self.render(mode='text')}\n\n"
            f"LEGAL ACTIONS:\n{all_actions}.\n\n"
            f"INSTRUCTIONS:\n{instructions}"
            )
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return prompt

    def get_all_actions(self):
        return self._get_legal_actions(self.current_player)

    def _get_legal_actions(self, player_id):
        legal_actions = dict()
        actions = self.state.legal_actions(player_id)
        for a in actions:
            legal_actions[a] = self._action_to_string(player_id, a)
        return legal_actions

    def _action_to_string(self, player_id, action):
        mark = "X" if player_id == 0 else "O"
        row = action // 3
        column = action % 3
        return f"{mark}({row},{column})"

    def _string_to_action(self, action_str):
        mark = action_str[0]
        row = int(action_str[2])
        column = int(action_str[4])
        return row * 3 + column

    def _get_info(self):
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
            return {"returns": returns, "winner": winner}
        else:
            return {}

    def _render_text(self):
        """Render the game state as text."""
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
        prompt = env.get_prompt()
        print("-"*100)
        print(f"System prompt: \n{prompt[0]['content']}")
        print(f"User prompt: \n{prompt[1]['content']}")
        print("-"*100)
        action = random.choice(list(env.get_all_actions().values()))
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
        api_key="token-reasoning-llm-zheyue-123",
    )

    # Call LLM to get action based on instruction
    def get_action_from_llm(prompt):
        response = client.chat.completions.create(
            model="/mnt/public/algm/models/Meta-Llama-3-8B-Instruct",
            messages=prompt
        )
        raw_content = response.choices[0].message.content
        try:
            action_json = json.loads(raw_content)
            return action_json["action"], raw_content
        except Exception as e:
            print("Failed to parse LLM response:", raw_content)
            raise e

    env = TicTacToe()
    num_steps = 5  # Number of steps to simulate
    from tqdm import tqdm
    env.reset()
    for step in tqdm(range(num_steps)):
        prompt = env.get_prompt()
        print(f"========== Step {step + 1} ==========")
        print(f"System prompt: \n{prompt[0]['content']}")
        print(f"User prompt: \n{prompt[1]['content']}")
        action, raw_content = get_action_from_llm(prompt)
        print(f"LLM Response: \n{raw_content}")
        print(f"LLM Action: {action}")
        observations, rewards, done, info = env.step(action)
        print(f"observations: \n{observations}")
        print(f"rewards: {rewards}")
        print(f"done: {done}")
        print(f"info: {info}")
