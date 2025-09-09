from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class HanabiConfig:
    seed: int = 42
    render_mode: str = "text"
    built_in_opponent: str = "none"
    opponent_first_move: bool = False
    include_opponent_turn: str = "action"

    # game config
    players: int = 2
    colors: int = 2
    ranks: int = 2
    hand_size: int = 3
    max_information_tokens: int = 3
    max_life_tokens: int = 3
    history_size: int = 4
