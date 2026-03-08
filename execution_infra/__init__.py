"""
Optimal Execution RL Environment package.

Quick usage
-----------
    from execution_infra import ExecutionEnv, EnvConfig

    env = ExecutionEnv()                        # synthetic market (default)
    env = ExecutionEnv(EnvConfig(n_steps=30))   # synthetic, custom params

    from execution_infra import AbidesReplayEnv
    env = AbidesReplayEnv(npz_path="data/features.npz")  # real ABIDES data
"""

from execution_infra.config import EnvConfig
from execution_infra.market_sim import MarketSimulator, LOBSnapshot
from execution_infra.execution_env import ExecutionEnv
from execution_infra.abides_replay_env import AbidesReplayEnv

__all__ = [
    "EnvConfig",
    "MarketSimulator",
    "LOBSnapshot",
    "ExecutionEnv",
    "AbidesReplayEnv",
]
