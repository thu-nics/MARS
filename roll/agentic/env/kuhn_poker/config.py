from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class KuhnPokerConfig:
    seed: int = 42
    render_mode: str = "text"
    built_in_opponent: str = "cfr"
    opponent_first_move: bool = False
    include_opponent_turn: str = "action"
    
    # mcts config
    uct_c: float = 2.0                   
    max_simulations: int = 100
    rollout_count: int = 10
    
    # cfr config
    cfr_iterations: int = 1000