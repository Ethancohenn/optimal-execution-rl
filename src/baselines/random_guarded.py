from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow running as either `python -m src.baselines.random_guarded` or
# `python src/baselines/random_guarded.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra.abides_replay_env import AbidesReplayEnv
from src.common.actions import DEFAULT_ACTION_SPEC, action_to_qty
from src.common.logger import RunLogger
from src.envs.stub_env import StubExecutionEnv


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("random_guarded_%Y%m%d_%H%M%S")
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


def _closest_discrete_action(remaining: float, max_trade_size: float, target_qty: float) -> int:
    candidates = [
        action_to_qty(a, remaining, max_trade_size, spec=DEFAULT_ACTION_SPEC)
        for a in range(len(DEFAULT_ACTION_SPEC.fractions))
    ]
    diffs = [abs(q - target_qty) for q in candidates]
    return int(diffs.index(min(diffs)))


def _guarded_random_qty(
    remaining: float,
    step_idx: int,
    horizon: int,
    spread_norm: float,
    rng: np.random.Generator,
    rand_low: float,
    rand_high: float,
    urgency_boost: float,
    urgency_power: float,
    spread_sensitivity: float,
    catchup_steps: int,
) -> float:
    if remaining <= 0:
        return 0.0

    steps_left = max(1, int(horizon) - int(step_idx))
    if steps_left == 1:
        return float(remaining)

    # Become more aggressive late in the episode.
    progress = float(step_idx + 1) / float(max(1, horizon))
    urgency = progress ** max(0.0, float(urgency_power))

    # If spread is wide, be slightly less aggressive; if tight, slightly more.
    spread_norm = max(1e-3, float(spread_norm))
    spread_scale = float(np.clip(spread_norm ** (-float(spread_sensitivity)), 0.75, 1.25))

    # Phase 1: genuinely random sizing on remaining inventory.
    # rand_low/high are fractions of *remaining*.
    if steps_left > max(1, int(catchup_steps)):
        low_frac = float(np.clip(rand_low, 0.0, 1.0))
        high_frac = float(np.clip(rand_high + float(urgency_boost) * urgency, 0.0, 1.0))
        if high_frac < low_frac:
            high_frac = low_frac
        frac = float(rng.uniform(low_frac, high_frac))
        target = remaining * frac * spread_scale
        return float(np.clip(target, 0.0, remaining))

    # Phase 2: randomized catch-up near deadline so completion is likely.
    catchup_qty = remaining / float(steps_left)
    jitter = float(rng.uniform(0.6, 1.8))
    raw_target = catchup_qty * jitter * spread_scale * (1.0 + 0.5 * urgency)
    floor = 0.5 * catchup_qty
    return float(np.clip(raw_target, floor, remaining))


def run_random_guarded(args: argparse.Namespace) -> str:
    if args.use_abides:
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.T,
            total_inventory=int(args.Q0),
            n_actions=len(DEFAULT_ACTION_SPEC.fractions),
            split=args.split,
        )
        print(f"[random_guarded] Using AbidesReplayEnv  npz={args.npz_path} (split={args.split})")
        max_trade_size = float(args.Q0)
    else:
        # Let the random policy choose from a full inventory-sized action cap.
        env = StubExecutionEnv(T=args.T, Q0=args.Q0, max_trade_size=args.Q0, seed=args.seed)
        print("[random_guarded] Using synthetic StubExecutionEnv")
        max_trade_size = float(args.Q0)

    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")
    rng = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0
        remaining = float(args.Q0)

        while not done:
            spread_norm = float(obs[3]) if obs is not None and len(obs) >= 4 else 1.0
            target_qty = _guarded_random_qty(
                remaining=remaining,
                step_idx=step_idx,
                horizon=int(args.T),
                spread_norm=spread_norm,
                rng=rng,
                rand_low=args.rand_low,
                rand_high=args.rand_high,
                urgency_boost=args.urgency_boost,
                urgency_power=args.urgency_power,
                spread_sensitivity=args.spread_sensitivity,
                catchup_steps=args.catchup_steps,
            )

            if hasattr(env, "step_with_qty"):
                next_obs, reward, terminated, truncated, info = env.step_with_qty(int(round(target_qty)))
                action = -1  # direct-quantity mode
            else:
                action = _closest_discrete_action(
                    remaining=remaining,
                    max_trade_size=max_trade_size,
                    target_qty=target_qty,
                )
                next_obs, reward, terminated, truncated, info = env.step(action)

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
                    "is_twap": 0,
                    "executed_qty": executed_qty,
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": step_is,
                }
            )
            step_idx += 1
            obs = next_obs

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_is": ep_is,
                "completed": int(remaining <= 1e-6),
                "steps": step_idx,
                "epsilon": 0.0,
                "strategy": "random_guarded",
                "is_twap": 0,
            }
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a guarded-random liquidation baseline on an execution env."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run directory before writing.")
    parser.add_argument(
        "--use-abides",
        action="store_true",
        help="Use AbidesReplayEnv (real ABIDES data) instead of StubExecutionEnv.",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data/features.npz",
        help="Path to features.npz produced by the feature extraction pipeline.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Train/test split of the data file (first 80%% or last 20%%). Default: test.",
    )

    # Smart-random controls.
    parser.add_argument(
        "--rand-low",
        type=float,
        default=0.02,
        help="Lower bound random fraction of remaining inventory (early/mid episode).",
    )
    parser.add_argument(
        "--rand-high",
        type=float,
        default=0.30,
        help="Upper bound random fraction of remaining inventory (early/mid episode).",
    )
    parser.add_argument("--urgency-boost", type=float, default=0.6, help="Extra aggressiveness near deadline.")
    parser.add_argument("--urgency-power", type=float, default=1.5, help="Curvature of urgency ramp.")
    parser.add_argument(
        "--spread-sensitivity",
        type=float,
        default=0.35,
        help="How much to reduce size when spread is wide.",
    )
    parser.add_argument(
        "--catchup-steps",
        type=int,
        default=6,
        help="Number of final steps using randomized catch-up logic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_path = run_random_guarded(parse_args())
    print(f"Saved random_guarded baseline artifacts to: {run_path}")
