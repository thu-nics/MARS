from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class ConnectFourConfig:
    seed: int = 42
    render_mode: str = "text"
    built_in_opponent: str = "mcts"
    opponent_first_move: bool = False
    include_opponent_turn: str = "full"
    
    # mcts config
    uct_c: float = 2.0                   
    max_simulations: int = 100
    rollout_count: int = 10
