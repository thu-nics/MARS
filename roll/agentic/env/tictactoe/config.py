from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class TicTacToeConfig:
    seed: int = 42
    render_mode: str = "text"
    random_opponent: bool = True