"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""

# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
# from .bandit.config import BanditEnvConfig
# from .bandit.env import BanditEnv
# from .countdown.config import CountdownEnvConfig
# from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
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
# from .metamathqa.env import MetaMathQAEnv
# from .metamathqa.config import MetaMathQAEnvConfig


REGISTERED_ENVS = {
    # "bandit": BanditEnv,
    # "countdown": CountdownEnv,
    "sokoban": SokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    "tictactoe": TicTacToe,
    "hanabi": Hanabi,
    "connect_four": ConnectFour,
    "kuhn_poker": KuhnPoker,
    "leduc_poker": LeducPoker,
    # 'alfworld': AlfredTXTEnv,
    # "metamathqa": MetaMathQAEnv,
}

REGISTERED_ENV_CONFIGS = {
    # "bandit": BanditEnvConfig,
    # "countdown": CountdownEnvConfig,
    "sokoban": SokobanEnvConfig,
    "frozen_lake": FrozenLakeEnvConfig,
    "tictactoe": TicTacToeConfig,
    "hanabi": HanabiConfig,
    "connect_four": ConnectFourConfig,
    "kuhn_poker": KuhnPokerConfig,
    "leduc_poker": LeducPokerConfig,
    # 'alfworld': AlfredEnvConfig,
    # "metamathqa": MetaMathQAEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig

    REGISTERED_ENVS["webshop"] = WebShopEnv
    REGISTERED_ENV_CONFIGS["webshop"] = WebShopEnvConfig
except Exception as e:
    pass
