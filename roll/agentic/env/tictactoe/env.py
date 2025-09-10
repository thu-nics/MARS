import matplotlib.pyplot as plt
import numpy as np
import os
import random
from roll.agentic.env.tictactoe.config import TicTacToeConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv
from textwrap import dedent
import json
from typing import Optional, Dict, Any
import re
from PIL import Image


class TicTacToe(BaseDiscreteActionEnv):
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(self, config: TicTacToeConfig = TicTacToeConfig()):
        # Using mappings directly from config
        self.config = config
        self.render_mode = config.render_mode
        self.built_in_opponent = config.built_in_opponent
        self.opponent_player = config.opponent_player
        self.include_opponent_turn = config.include_opponent_turn

        BaseDiscreteActionEnv.__init__(self)

        import pyspiel
        self._env = pyspiel.load_game("tic_tac_toe")
        self.state = None

        if self.built_in_opponent == "mcts":
            from open_spiel.python.algorithms import mcts
            random_state = np.random.RandomState(config.seed)
            evaluator = mcts.RandomRolloutEvaluator(config.rollout_count, random_state)
            self.mcts_bot = mcts.MCTSBot(
                self._env,
                config.uct_c,
                config.max_simulations,
                evaluator,
                solve=False,
                random_state=random_state,
            )

    @property
    def current_player(self):
        if self.state is None:
            return 0
        return self.state.current_player()

    def reset(self, seed: Optional[int] = 0):
        try:
            with all_seed(seed):
                self.state = self._env.new_initial_state()
                initial_observation = {
                        'observation': self.render(),
                        'legal_actions': self.get_all_actions(),
                }
                execute_results = []
                if self.built_in_opponent != "none":
                    done = self.state.is_terminal()
                    while self.current_player == self.opponent_player and not done:
                        current_player = self.current_player
                        opponent_action = self._opponent_step()
                        observation, rewards, done, info = self._step(opponent_action)
                        execute_results.append({
                            'current_player': current_player,
                            'action': self._action_to_string(current_player, opponent_action),
                            'rewards': rewards,
                            'done': done,
                            'info': info,
                            'next_player': self.current_player,
                            'observation': observation,
                            'legal_actions': self.get_all_actions(),
                        })
                return initial_observation, execute_results
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else 0
            return self.reset(next_seed)

    def step(self, action):
        execute_results = []
        current_player = self.current_player
        observation, rewards, done, info = self._step(action)

        execute_results.append({
            'current_player': current_player,
            'action': self._action_to_string(current_player, action),
            'rewards': rewards,
            'done': done,
            'info': info,
            'next_player': self.current_player,
            'observation': observation,
            'legal_actions': self.get_all_actions(),
        })
        # If chose to play with built-in opponent, we need to let the opponent take action
        if self.built_in_opponent != "none":
            while self.current_player == self.opponent_player and not done:
                current_player = self.current_player
                opponent_action = self._opponent_step()
                observation, rewards, done, info = self._step(opponent_action)
                execute_results.append({
                    'current_player': current_player,
                    'action': self._action_to_string(current_player, opponent_action),
                    'rewards': rewards,
                    'done': done,
                    'info': info,
                    'next_player': self.current_player,
                    'observation': observation,
                    'legal_actions': self.get_all_actions(),
                })
        return execute_results

    def _step(self, action):
        if isinstance(action, str):
            action = self._string_to_action(action)
        if self.state is None or self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.state.apply_action(action)
        observation = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info()
        if info != {}:
            rewards = [0.1 if reward == 0 else reward for reward in rewards]
        return observation, rewards, done, info

    def _opponent_step(self):
        if self.built_in_opponent == "random":
            action = random.choice(list(self.get_all_actions().values()))
        elif self.built_in_opponent == "mcts":
            action = self.mcts_bot.step(self.state)
        else:
            raise ValueError(f"Invalid built-in opponent: {self.built_in_opponent}")
        # print(f"Built-in {self.built_in_opponent} opponent taking action: {self._action_to_string(self.current_player, action)}")
        return action

    def get_prompt(self, mode="prefix", think=True, player_id=0):
        if mode == "prefix":
            prefix_prompt = self._get_prefix_prompt(think, player_id)
            return prefix_prompt
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True, player_id=0):
        system_prompt = "You are an AI agent that makes optimal decisions to win in the game of tic-tac-toe."
        rules = (
            "1. Tic-tac-toe is a two-player board game played on a three-by-three grid. "
            "The grid is 0-indexed, where (0,0) is the top-left corner and (2,2) is the bottom-right corner.\n"
            "2. Two players take turns placing their marks X and O in empty cells of the grid.\n"
            "3. The player who first places three of their marks in a horizontal, vertical, or diagonal line wins.\n"
            "4. If all cells are filled and no player wins, the game ends in a draw."
        )
        mark = "O" if player_id == 1 else "X"
        opponent_mark = "O" if mark == "X" else "X"
        information = (
            f"1. Your mark is {mark}. You are competing with another player controlling the mark {opponent_mark}.\n"
            "2. In each of your turns:\n"
            "   a. The game state demonstrates the current board with a three-line text grid, where 'X' and 'O' are the marks of the two players, and '_' represents empty cells.\n"
            "   b. You need to chose an action to place your mark in an empty cell, based on the given game state and the history of your decisions.\n"
            f"   c. All legal actions for the current turn are provided in the format of `<{mark}({{row}},{{column}})>`, where `{mark}` is your mark, "
            "and {row} and {column} are integers indicating the row and column of the cell to place your mark."
        )
        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = f"<answer><{mark}(0,0)></answer>"
        instructions = (
            f"Always choose only one action from the legal actions and output `{FORMAT_PROMPT}` with no extra text after you finish the thinking process. "
            f"For example, `{FORMAT_PROMPT_EXAMPLE}`. "
            "Strictly follow the above format and keep your thinking process concise. Responses that do not follow the format will result in immediate loss of the game."
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

    def _action_to_string(self, player_id, action):
        if isinstance(action, str):
            return action
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
            "player_0_lose_for_wrong_format": 0,
            "player_1_lose_for_wrong_format": 0,
            "player_0_lose_for_overlong_response": 0,
            "player_1_lose_for_overlong_response": 0,
            "player_0_success": winner == 0,
            "player_1_success": winner == 1,
            "draw": winner == -1,
        }

    def get_losing_state(self, player_id: int=0, overlong_response: bool=False):
        observation = self.render()
        done = True
        if player_id == 0:
            reward = [-1 - 10, 0]
            info = {
                "player_0_return": -1,
                "player_1_return": 1,
                "winner": 1,
                "player_0_lose_for_wrong_format": 1,
                "player_1_lose_for_wrong_format": 0,
                "player_0_lose_for_overlong_response": 1 if overlong_response else 0,
                "player_1_lose_for_overlong_response": 0,
                "player_0_success": False,
                "player_1_success": True,
                "draw": False,
            }
        else:
            reward = [0, -1 - 10]
            info = {
                "player_0_return": 1,
                "player_1_return": -1,
                "winner": 0,
                "player_0_lose_for_wrong_format": 0,
                "player_1_lose_for_wrong_format": 1,
                "player_0_lose_for_overlong_response": 0,
                "player_1_lose_for_overlong_response": 1 if overlong_response else 0,
                "player_0_success": True,
                "player_1_success": False,
                "draw": False,
            }
        execute_results = [{
            'current_player': player_id,
            'action': '',
            'rewards': reward,
            'done': done,
            'info': info,
            'next_player': None,
            'observation': None,
            'legal_actions': None,
        }]
        return execute_results

    def render(self, mode: str = "text"):
        if mode == "text":
            return self._render_text()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
                if piece == ".":
                    row.append("_")
                elif piece == "x":
                    row.append("X")
                elif piece == "o":
                    row.append("O")
                else:
                    row.append(piece)
            text_repr.append("".join(row))
        return "\n".join(text_repr)

    def _render_rgb_array(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.invert_yaxis()
        for x in range(1, 3):
            ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color="black", linewidth=2)
        for y in range(1, 3):
            ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color="black", linewidth=2)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["0", "1", "2"])
        ax.set_yticklabels(["0", "1", "2"])
        if self.state is not None:
            board = np.array([list(line) for line in str(self.state).strip().split("\n")])
            for i in range(3):
                for j in range(3):
                    piece = board[i][j]
                    if piece != ".":
                        color = "red" if piece == "x" else "blue"
                        ax.text(j, i, piece.upper(), fontsize=30, ha="center", va="center", color=color)
        ax.set_aspect("equal")

        # convert matplotlib figure to numpy array, avoid file I/O
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        image = Image.fromarray(buf)
        plt.close(fig)
        return image

    def close(self):
        """Close the environment."""
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()


if __name__ == "__main__":
    # Basic unit test
    print("-" * 100)
    print("Basic unit test:")
    print("-" * 100)
    env = TicTacToe()
    
    player_0_returns = []
    player_1_returns = []
    for i in range(100):
        print('-' * 100)
        print(f'Episode {i}')
        print('-' * 100)
        prefix_prompt = env.get_prompt(mode="prefix")
        print(f"System prompt: \n{prefix_prompt['system']}")
        print(f"User prompt: \n{prefix_prompt['user']}")

        seed = i
        initial_observation, execute_results = env.reset(seed)
        observation = execute_results[-1]['observation'] if execute_results else initial_observation['observation']
        legal_actions = execute_results[-1]['legal_actions'] if execute_results else initial_observation['legal_actions']
        done = False
        while not done:
            print(f"observation: \n{observation}")
            print(f"legal actions: \n{legal_actions}")
            action = random.choice(list(legal_actions.values()))
            print(f"Player {env.current_player} taking action: {action}")

            execute_result = env.step(action)
            rewards = execute_result[-1]['rewards']
            done = execute_result[-1]['done']
            info = execute_result[-1]['info']
            print(f"rewards: {rewards}")
            print(f"done: {done}")
            print(f"info: {info}")
            print("-" * 100)

            observation = execute_result[-1]['observation']
            legal_actions = execute_result[-1]['legal_actions']
            
        player_0_returns.append(info['player_0_return'])
        player_1_returns.append(info['player_1_return'])
    print("-" * 100)
    print("player 0 returns: ", np.mean(player_0_returns))
    print("player 1 returns: ", np.mean(player_1_returns))
    print("-" * 100)
