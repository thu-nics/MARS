from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict


@dataclass
class TicTacToeConfig:
    seed: int = 42
    render_mode: str = "text"
    action_lookup: Optional[Dict[int, str]] = field(
        default_factory=lambda: {
            0: "Top-Left",
            1: "Top-Center",
            2: "Top-Right",
            3: "Center-Left",
            4: "Center",
            5: "Center-Right",
            6: "Bottom-Left",
            7: "Bottom-Center",
            8: "Bottom-Right",
        }
    )
