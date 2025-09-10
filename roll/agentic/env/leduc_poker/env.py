import matplotlib.pyplot as plt
import numpy as np
import os
import random
from roll.agentic.env.leduc_poker.config import LeducPokerConfig
from roll.agentic.utils import all_seed
from roll.agentic.env.base import BaseDiscreteActionEnv
from textwrap import dedent
import json
from itertools import permutations
from typing import Optional, Dict, Any
import re
from PIL import Image
import warnings


class LeducPoker(BaseDiscreteActionEnv):
    """Leduc Poker game environment using OpenSpiel."""

    def __init__(self, config: LeducPokerConfig = LeducPokerConfig()):
        # Using mappings directly from config
        self.config = config
        self.render_mode = config.render_mode
        self.built_in_opponent = config.built_in_opponent
        self.opponent_player = config.opponent_player
        self.include_opponent_turn = config.include_opponent_turn

        BaseDiscreteActionEnv.__init__(self)

        import pyspiel
        self._env = pyspiel.load_game("leduc_poker")
        self.state = None
        self.public_card = None
        self.bets = [1, 1]  # Initial ante for both players

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
        elif self.built_in_opponent == "cfr":
            from open_spiel.python.algorithms import cfr
            import pickle
            import gzip
            import os
            self.cfr_solver = cfr.CFRSolver(self._env)
            cfr_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfr.pkl.gz")
            if os.path.exists(cfr_checkpoint_path):
                with gzip.open(cfr_checkpoint_path, "rb") as f:
                    self.cfr_avg_policy = pickle.load(f)
            else:
                for _ in range(config.cfr_iterations):
                    self.cfr_solver.evaluate_and_update_policy()
                self.cfr_avg_policy = self.cfr_solver.average_policy()
        elif self.built_in_opponent == "ne":
            import pickle
            import gzip
            import os
            ne_checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ne.pkl.gz")
            with gzip.open(ne_checkpoint_path, "rb") as f:
                self.ne_policy_lookup = pickle.load(f)

    @property
    def current_player(self):
        if self.state is None or self.state.is_terminal():
            return 0
        return self.state.current_player()

    def reset(self, seed: Optional[int] = 0):
        """Reset the environment with given seed.
        Leduc Poker has 6 cards: J, Q, K (suits: Hearts, Spades)
        """
        try:
            with all_seed(seed):
                # Initialize game state and deal cards
                deck = {0: "J", 1: "J", 2: "Q", 3: "Q", 4: "K", 5: "K"}
                triple_dict = {}
                for actions in permutations(deck.keys(), 3):
                    cards = tuple(deck[a] for a in actions)
                    if cards[0] == cards[1] == cards[2]:
                        continue
                    if cards not in triple_dict:
                        triple_dict[cards] = actions
                action_triples = list(triple_dict.values())
                card_0, card_1, self.public_card = action_triples[seed % len(action_triples)]

                self.state = self._env.new_initial_state()
                self.state.apply_action(card_0)
                self.state.apply_action(card_1)

                # self.bets = [1, 1]  # Both players place blind ante
                self.bets = [100 - self.state.money()[0], 100 - self.state.money()[1]]
                
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
        if self.state.is_chance_node():
            self.state.apply_action(self.public_card)
        
        self.bets = [100 - self.state.money()[0], 100 - self.state.money()[1]]
        observation = self.render()
        rewards = self.state.rewards()
        done = self.state.is_terminal()
        info = self._get_info()
        return observation, rewards, done, info

    def _opponent_step(self):
        if self.built_in_opponent == "random":
            action = random.choice(list(self.get_all_actions().values()))
        elif self.built_in_opponent == "mcts":
            action = self.mcts_bot.step(self.state)
        elif self.built_in_opponent == "cfr":
            state_policy = self.cfr_avg_policy.action_probabilities(self.state)
            actions = list(state_policy.keys())
            probabilities = list(state_policy.values())
            action = int(np.random.choice(actions, p=probabilities))
        elif self.built_in_opponent == "ne":
            state_key = self._get_state_key()
            state_policy = self.ne_policy_lookup[state_key]
            action = int(np.random.choice(range(3), p=state_policy))
        else:
            raise ValueError(f"Invalid built-in opponent: {self.built_in_opponent}")
        # print(f"Built-in {self.built_in_opponent} opponent taking action: {self._action_to_string(self.current_player, action)}")
        return action

    def _get_state_key(self):
        info_state = self.state.information_state_tensor(self.current_player)
        
        cards = ''
        # Show current player's card
        deck = ["J", "J", "Q", "Q", "K", "K"]
        if len(info_state) >= 8:
            card_idx = np.argmax(info_state[2:8])  # Private card is at info_state[2:8]
            cards += deck[card_idx]
        
        # Show public card if revealed
        if len(info_state) >= 14 and np.sum(info_state[8:14]) > 0:
            public_card_idx = np.argmax(info_state[8:14])  # Public card is at info_state[8:14]
            cards += deck[public_card_idx]
        else:
            cards += '_'
        
        # Process betting rounds using observation tensor
        action_space = ["<CALL>", "<RAISE>"]
        
        # First betting round (info_state[14:22])
        turn1 = ""
        if len(info_state) >= 22:
            num_turns = int(np.sum(info_state[14:22]))
            for i in range(num_turns):
                action = action_space[np.argmax(info_state[14 + 2 * i:16 + 2 * i])]
                turn1 += 'c' if action == "<CALL>" else 'r'
        
        # Second betting round (info_state[22:30])
        turn2 = ""
        if len(info_state) >= 30:
            num_turns = int(np.sum(info_state[22:30]))
            for i in range(num_turns):
                action = action_space[np.argmax(info_state[22 + 2 * i:24 + 2 * i])]
                turn2 += 'c' if action == "<CALL>" else 'r'

        return f"[{cards}]_[{turn1}]_[{turn2}]"

    def get_prompt(self, mode="prefix", think=True, player_id=0):
        if mode == "prefix":
            prefix_prompt = self._get_prefix_prompt(think, player_id)
            return prefix_prompt
        else:
            raise ValueError(f"Invalid prompt mode: {mode}")

    def _get_prefix_prompt(self, think=True, player_id=0):
        system_prompt = "You are an AI agent that makes optimal decisions to win in the game of Leduc Poker."
        rules = (
            "1. Leduc poker is a two-player card game. The deck includes only six cards: two pairs of King (K), Queen (Q), and Jack (J).\n"
            "2. At the start of each game, both player_0 and player_1 place 1 chip into the pot as a blind ante.\n"
            "3. Each player is dealt one private card from the deck, and the remaining cards are set aside unseen.\n"
            "4. The game has two betting rounds. When the first round ends, one public card from the remaining cards of the deck is revealed to both players.\n"
            "5. The two players take turns acting in the betting rounds, both starting with player_0. A player can choose to:\n"
            "    a. <FOLD>: stop betting and the other player wins the pot.\n"
            "    b. <CALL>: match the current bet. If no bet has been made in the current round, this is equivalent to checking.\n"
            "    c. <RAISE>: first match the current bet and then add `n` chips to the bet, where `n=2` in the first round and `n=4` in the second round. "
            "If no bet has been made in the current round, this is equivalent to betting `n` chips.\n"
            "6. A maximum of two <RAISE>s are allowed in each round. Each round ends when both players have acted and their bets are equal.\n"
            "7. If a player chooses to <FOLD>, the other player wins the pot.\n"
            "8. If neither player chooses to <FOLD>, the second round ends with a showdown:\n"
            "    a. If a player has a pair (private card = public card), the player wins the pot.\n"
            "    b. If neither player has a pair, the player with the higher card (K > Q > J) wins the pot.\n"
            "    c. If two players have the same card, the players split the pot."
        )
        information = (
            f"1. You are player_{player_id}. You are competing with player_{1-player_id}.\n"
            "2. In each of your turns:\n"
            "   a. The game state shows your private card, public card (if revealed), and the betting history.\n"
            "   b. You need to choose an action based on your cards and the current game state.\n"
            "   c. All legal actions for the current turn are provided in the format of `<FOLD>`, `<CALL>`, or `<RAISE>`."
        )
        FORMAT_PROMPT = "<answer>{your chosen action}</answer>"
        FORMAT_PROMPT_EXAMPLE = "<answer><CALL></answer>"
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
        if action == 0:
            return "<FOLD>"
        elif action == 1:
            return "<CALL>"
        else:  # action == 2
            return "<RAISE>"

    def _string_to_action(self, action_str):
        action_str = action_str.strip().upper()
        if action_str == "<FOLD>":
            return 0
        elif action_str == "<CALL>":
            return 1
        elif action_str == "<RAISE>":
            return 2
        else:
            # Try to extract from different formats
            if "FOLD" in action_str:
                return 0
            elif "CALL" in action_str:
                return 1
            elif "RAISE" in action_str:
                return 2
            else:
                raise ValueError(f"Invalid action string: {action_str}")

    def _get_info(self):
        if self.state is None:
            return {}
        
        # If game just started, return card information
        if len(self.state.history()) == 2 and not self.state.is_terminal():
            deck = ["J", "J", "Q", "Q", "K", "K"]
            card_0_idx = self.state.history()[0]
            card_1_idx = self.state.history()[1]
            return {
                "card_0": deck[card_0_idx],
                "card_1": deck[card_1_idx],
                "public_card": deck[self.public_card],
                "total_pot": float(sum(self.bets)),
                "player_0_bet": float(self.bets[0]),
                "player_1_bet": float(self.bets[1]),
            }
        
        # If game is terminal, return final results
        if self.state.is_terminal():
            returns = self.state.returns()
            winner = int(np.argmax(returns)) if returns[0] != returns[1] else -1
            deck = ["J", "J", "Q", "Q", "K", "K"]
            
            card_0_idx = self.state.history()[0]
            card_1_idx = self.state.history()[1]

            return {
                "card_0": deck[card_0_idx],
                "card_1": deck[card_1_idx],
                "public_card": deck[self.public_card],
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
                "total_pot": float(sum(self.bets)),
                "player_0_bet": float(self.bets[0]),
                "player_1_bet": float(self.bets[1]),
            }
        else:
            # Game in progress
            info = {
                "total_pot": float(sum(self.bets)),
                "player_0_bet": float(self.bets[0]),
                "player_1_bet": float(self.bets[1]),
            }
            
            return info

    def get_losing_state(self, player_id: int=0, overlong_response: bool=False):
        observation = self.render()
        done = True
        if player_id == 0:
            reward = [-10, 0]
            info = {
                "player_0_return": -self.bets[0],
                "player_1_return": self.bets[0],
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
            reward = [0, -10]
            info = {
                "player_0_return": self.bets[1],
                "player_1_return": -self.bets[1],
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
        """
        Render the game state as text.
            * info_state[0:2]: player_id 
            * info_state[2:8]: one-hot encode of self card (J1, J2, Q1, Q2, K1, K2)
            * info_state[8:14]: one-hot encode of public card (J1, J2, Q1, Q2, K1, K2)
            * info_state[14:22]: one-hot encode of first round actions (call, raise)
            * info_state[22:30]: one-hot encode of second round actions (call, raise)
        """

        if self.state is None:
            return "Game not started"

        info_state = self.state.information_state_tensor(self.current_player)
        history = ["1. Blind ante: both player_0 and player_1 place 1 chip into the pot."]
        round_str = "first" if np.sum(info_state[8:14]) == 0 else "second"
        
        # Show current player's card
        deck = ["J", "J", "Q", "Q", "K", "K"]
        if len(info_state) >= 8:
            card_idx = np.argmax(info_state[2:8])  # Private card is at info_state[2:8]
            current_card = deck[card_idx]
            history.append(f"2. Deal: your card is {current_card}.")
        
        # Show public card if revealed
        if len(info_state) >= 14 and np.sum(info_state[8:14]) > 0:
            public_card_idx = np.argmax(info_state[8:14])  # Public card is at info_state[8:14]
            public_card = deck[public_card_idx]
            history.append(f"3. Public card revealed: {public_card}.")
        
        # Process betting rounds using observation tensor
        action_space = ["<CALL>", "<RAISE>"]
        
        # First betting round (info_state[14:22])
        if len(info_state) >= 22:
            bets = np.zeros(3, dtype=int)  # player_0, player_1, highest bet
            num_turns = int(np.sum(info_state[14:22]))
            if num_turns > 0:
                history.append("4. First betting round:")
            for i in range(num_turns):
                player_id = i % 2
                action = action_space[np.argmax(info_state[14 + 2 * i:16 + 2 * i])]
                bets[-1] += 2 if action == "<RAISE>" else 0
                add_chips = bets[-1] - bets[player_id]
                bets[player_id] = bets[-1]
                history.append(f"    {chr(97 + i)}. Turn {i + 1}: player_{player_id} chooses to {action} (+{add_chips} chips).")
        
        # Second betting round (info_state[22:30])
        if len(info_state) >= 30:
            bets = np.zeros(3, dtype=int)  # player_0, player_1, highest bet
            num_turns = int(np.sum(info_state[22:30]))
            if num_turns > 0:
                history.append("5. Second betting round:")
            for i in range(num_turns):
                player_id = i % 2
                action = action_space[np.argmax(info_state[22 + 2 * i:24 + 2 * i])]
                bets[-1] += 4 if action == "<RAISE>" else 0
                add_chips = bets[-1] - bets[player_id]
                bets[player_id] = bets[-1]
                history.append(f"    {chr(97 + i)}. Turn {i + 1}: player_{player_id} chooses to {action} (+{add_chips} chips).")
        
        return "\n".join(history)

    def _render_rgb_array(self):
        warnings.warn("Leduc Poker does not support image rendering yet.")
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()


if __name__ == "__main__":
    # Basic unit test
    print("-" * 100)
    print("Basic unit test:")
    print("-" * 100)
    env = LeducPoker()
    
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