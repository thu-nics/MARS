import matplotlib.pyplot as plt
import numpy as np
import os
import pyspiel
from .config import TicTacToeEnvConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv


class TicTacToe(BaseDiscreteActionEnv):
    """Tic-Tac-Toe game environment using OpenSpiel."""

    def __init__(self, config: TicTacToeEnvConfig = TicTacToeEnvConfig()):
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

    def step(self, action):
        if self.state.is_terminal():
            raise RuntimeError("Cannot apply action on a terminal state.")

        self.state.apply_action(action)
        observations = self.render()
        rewards = self.state.rewards()
        dones = self.state.is_terminal()
        info = self._get_info()

        return observations, rewards[self.current_player], dones[self.current_player], info

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

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys()) if self.ACTION_LOOKUP else []

    def _get_info(self):
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
            return {"returns": returns, "winner": winner}
        else:
            return {}

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
        pass