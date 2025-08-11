import numpy as np
import os
import pyspiel
import json
from open_spiel.python.algorithms import mcts


def state_to_text(state):
    """Render the game state as text."""
    text_repr = []
    for i in range(3):
        row = ''.join(state[i*3:(i+1)*3])
        text_repr.append(row)
    return "\n".join(text_repr)

def state_to_action_history(state):
    """Convert a board state to a list of action history."""
    x_history = []
    o_history = []
    for i in range(len(state)):
        if state[i] == 'X':
            x_history.append(i)
        elif state[i] == 'O':
            o_history.append(i)
    return x_history, o_history

def check_winner(board):
    """Checks if there is a winner on the board."""
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)             # Diagonals
    ]
    for i, j, k in win_conditions:
        if board[i] == board[j] == board[k] and board[i] != '_':
            return True
    return False

def is_board_full(board):
    """Checks if the board is full."""
    return '_' not in board

# A set to store all unique legal board states encountered.
legal_states = set()

def generate_states(board, player):
    """
    Recursively generates all legal states of Tic-Tac-Toe.
    """
    # Add the current board state to our set of legal states.
    legal_states.add(board)

    # If there is a winner or the board is full, no more moves can be made from this state.
    if check_winner(board) or is_board_full(board):
        return

    # Determine the next player.
    next_player = 'O' if player == 'X' else 'X'

    # Iterate through all empty cells to generate next possible states.
    for i in range(9):
        if board[i] == '_':
            # Create a new board with the player's move.
            new_board = list(board)
            new_board[i] = player
            new_board_tuple = tuple(new_board)

            # Recurse with the new board and the other player.
            generate_states(new_board_tuple, next_player)

def main():
    """
    Main function to generate and print all legal Tic-Tac-Toe states.
    """
    initial_board = ('_', '_', '_', '_', '_', '_', '_', '_', '_')
    generate_states(initial_board, 'X')

    print(f"Total number of unique legal states: {len(legal_states)}\n")

    # Sort the states for a consistent output order (optional).
    sorted_states = sorted(list(legal_states))

    env = pyspiel.load_game("tic_tac_toe")
    # value_table = json.load(open("./tictactoe_value_table.json", "r"))
    value_table = {}
    random_state = np.random.RandomState(0)
    evaluator = mcts.RandomRolloutEvaluator(10, random_state)
    mcts_bot = mcts.MCTSBot(
        env,
        2,
        1000,
            evaluator,
            solve=False,
            random_state=random_state,
        )
    
    for i, state in enumerate(sorted_states):
        state_str = state_to_text(state)
        if state_str in value_table:
            continue

        x_history, o_history = state_to_action_history(state)

        game_state = env.new_initial_state()
        while (x_history or o_history) and not game_state.is_terminal():
            if game_state.current_player() == 0:
                action = x_history.pop(0)
            else:
                action = o_history.pop(0)
            game_state.apply_action(action)

        while not game_state.is_terminal():
            action = mcts_bot.step(game_state)
            game_state.apply_action(action)

        returns = game_state.returns()
        value_table[state_str] = [returns[0], returns[1]]
        print(f"Log new state to value table: {i}/{len(sorted_states)} \n", state_str)
        json.dump(value_table, open("./tictactoe_value_table.json", "w"), indent=4)
        

if __name__ == "__main__":
    main()