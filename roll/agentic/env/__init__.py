"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""

from .tictactoe.config import TicTacToeConfig
from .tictactoe.env import TicTacToe
from .hanabi.config import HanabiConfig
from .hanabi.env import Hanabi
from .connect_four.config import ConnectFourConfig
from .connect_four.env import ConnectFour
from .kuhn_poker.config import KuhnPokerConfig
from .kuhn_poker.env import KuhnPoker
from .leduc_poker.config import LeducPokerConfig
from .leduc_poker.env import LeducPoker

REGISTERED_ENVS = {
    "tictactoe": TicTacToe,
    "hanabi": Hanabi,
    "connect_four": ConnectFour,
    "kuhn_poker": KuhnPoker,
    "leduc_poker": LeducPoker,
}

REGISTERED_ENV_CONFIGS = {
    "tictactoe": TicTacToeConfig,
    "hanabi": HanabiConfig,
    "connect_four": ConnectFourConfig,
    "kuhn_poker": KuhnPokerConfig,
    "leduc_poker": LeducPokerConfig,
}
