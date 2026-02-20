"""
Optimal Execution RL Environment package.

Quick usage
-----------
    from execution_infra import ExecutionEnv, EnvConfig

    env = ExecutionEnv()                     # default hyper-params
    env = ExecutionEnv(EnvConfig(n_steps=30)) # custom
"""

from execution_infra.config import EnvConfig
from execution_infra.market_sim import MarketSimulator, LOBSnapshot
from execution_infra.execution_env import ExecutionEnv

__all__ = [
    "EnvConfig",
    "MarketSimulator",
    "LOBSnapshot",
    "ExecutionEnv",
]
