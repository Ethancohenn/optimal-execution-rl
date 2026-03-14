from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv
from src.common.actions import DEFAULT_ACTION_SPEC
from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common
from execution_infra.abides_replay_env import AbidesReplayEnv


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix="twap",
    )


def run_twap(args: argparse.Namespace) -> str:
    if args.use_abides:
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.T,
            total_inventory=int(args.Q0),
            n_actions=len(DEFAULT_ACTION_SPEC.fractions),
            split=args.split,
        )
        print(f"[twap] Using AbidesReplayEnv  npz={args.npz_path} (split={args.split})")
    else:
        max_trade_size = args.Q0 / float(args.T)
        env = ExecutionEnv(
            EnvConfig(
                total_inventory=int(args.Q0),
                n_steps=int(args.T),
                force_liquidation=True,
                seed=args.seed,
            )
        )
        print("[twap] Using synthetic ExecutionEnv (stateful light LOB)")
    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    action_fractions = DEFAULT_ACTION_SPEC.fractions

    def _twap_action(remaining: float, target_qty: float) -> int:
        if remaining <= 0:
            return 0
        qtys = [f * remaining for f in action_fractions]
        diffs = [abs(q - target_qty) for q in qtys]
        return int(diffs.index(min(diffs)))

    for episode in range(args.episodes):
        _, reset_info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0
        remaining = float(args.Q0)
        target_qty = args.Q0 / args.T

        while not done:
            if hasattr(env, "step_with_qty"):
                _, reward, terminated, truncated, info = env.step_with_qty(int(round(target_qty)))
                action = -1
            else:
                action = _twap_action(remaining, target_qty)
                _, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            exec_price = float(info.get("exec_price", float("nan")))
            mid_price = float(info.get("mid_price", float("nan")))
            executed_qty = float(info.get("executed_qty", 0.0))
            if "step_is" in info:
                step_is = float(info["step_is"])
            elif executed_qty > 0 and not (
                exec_price != exec_price or mid_price != mid_price
            ):
                step_is = executed_qty * (mid_price - exec_price)
            else:
                step_is = -float(reward)
            ep_reward += float(reward)
            ep_is += step_is

            remaining = float(info.get("remaining_inventory", info.get("inventory_remaining", 0.0)))
            t_val = int(info.get("t", info.get("step", step_idx)))
            timestamp_ns = int(info.get("timestamp_ns", -1))

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": t_val,
                    "timestamp_ns": timestamp_ns,
                    "remaining_inventory": remaining,
                    "action": int(action),
                    "exec_price": exec_price,
                    "mid_price": mid_price,
                    "reward": float(reward),
                    "is_twap": 1,
                    "executed_qty": executed_qty,
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
                "completed": int(remaining <= 1e-6),
                "steps": step_idx,
                "epsilon": 0.0,
                "strategy": "twap",
                "is_twap": 1,
            }
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TWAP baseline on an execution env.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-abides", action="store_true")
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    return parser.parse_args()


if __name__ == "__main__":
    run_path = run_twap(parse_args())
    print(f"Saved TWAP artifacts to: {run_path}")
