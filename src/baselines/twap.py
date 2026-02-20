from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow running as either `python -m src.baselines.twap` or `python src/baselines/twap.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.actions import DEFAULT_ACTION_SPEC
from src.common.logger import RunLogger
from src.envs.stub_env import StubExecutionEnv


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("twap_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            try:
                shutil.rmtree(run_dir)
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot overwrite run directory '{run_dir}'. "
                    "Close files under this folder (editor/Explorer) or use a different --run-name."
                ) from exc
        else:
            raise ValueError(
                f"Run directory already exists: {run_dir}. "
                "Use --overwrite or choose a new --run-name."
            )
    return run_dir


def run_twap(args: argparse.Namespace) -> str:
    max_trade_size = args.Q0 / float(args.T)
    env = StubExecutionEnv(T=args.T, Q0=args.Q0, max_trade_size=max_trade_size, seed=args.seed)
    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")
    full_trade_action = len(DEFAULT_ACTION_SPEC.fractions) - 1

    for episode in range(args.episodes):
        _, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0

        while not done:
            action = full_trade_action
            _, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            step_is = float(info.get("step_is", float(info["executed_qty"]) * (float(info["mid_price"]) - float(info["exec_price"]))))
            ep_reward += float(reward)
            ep_is += step_is

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": int(info["t"]),
                    "remaining_inventory": float(info["remaining_inventory"]),
                    "action": int(action),
                    "exec_price": float(info["exec_price"]),
                    "mid_price": float(info["mid_price"]),
                    "reward": float(reward),
                    "is_twap": 1,
                    "executed_qty": float(info["executed_qty"]),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": step_is,
                }
            )
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_is": ep_is,
                "completed": int(env.remaining_inventory <= 1e-6),
                "steps": step_idx,
                "epsilon": 0.0,
                "strategy": "twap",
                "is_twap": 1,
            }
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TWAP baseline on StubExecutionEnv.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run directory before writing.")
    return parser.parse_args()


if __name__ == "__main__":
    run_path = run_twap(parse_args())
    print(f"Saved TWAP artifacts to: {run_path}")
