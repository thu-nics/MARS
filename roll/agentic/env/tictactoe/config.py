from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class TicTacToeConfig:
    seed: int = 42
    render_mode: str = "text"
    built_in_opponent: str = "mcts"
    opponent_first_move: bool = False
    
    # mcts config
    uct_c: float = 2.0                   
    max_simulations: int = 100
    rollout_count: int = 10