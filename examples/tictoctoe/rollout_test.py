# -*- encoding: utf-8 -*-
# File: rollout_test.py
# Description: None

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
from roll.agentic.env.tictactoe.env import TicTacToe, TicTacToeConfig


def init_env():
    config = TicTacToeConfig(
        seed=42,
        render_mode="text",
        random_opponent=True,
    )
    env = TicTacToe(config=config)
    return env


def main():
    env = init_env()
    _ = env.reset()
    print("Initial observation:", _)

    for _ in range(5):
        left_actions = list(env.get_all_actions().values())

        if not left_actions:
            print("No actions left, ending the rollout.")
            break

        action = random.choice(left_actions)
        obs, reward, terminated, info = env.step(action)
        print("Action taken:", action)
        print("Observation after action:", obs)
        print("Reward received:", reward)
        print("Terminated:", terminated)
        print("Info:", info)


if __name__ == "__main__":
    main()
