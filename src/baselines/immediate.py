from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow running as either `python -m src.baselines.immediate` or `python src/baselines/immediate.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra.abides_replay_env import AbidesReplayEnv
from execution_infra.light_lob_env import LightLOBEnv
from execution_infra.tiny_lob_env import TinyLOBEnv
from src.common.actions import DEFAULT_ACTION_SPEC
from src.common.logger import RunLogger
from src.envs.stub_env import StubExecutionEnv


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("immediate_%Y%m%d_%H%M%S")
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


def _resolve_env_name(args: argparse.Namespace) -> str:
    return "abides" if args.use_abides else str(args.env)


def run_immediate(args: argparse.Namespace) -> str:
    env_name = _resolve_env_name(args)
    if env_name == "abides":
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.T,
            total_inventory=int(args.Q0),
            n_actions=len(DEFAULT_ACTION_SPEC.fractions),
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            split=args.split,
        )
        print(f"[immediate] Using AbidesReplayEnv  npz={args.npz_path} (split={args.split})")
    elif env_name == "stub":
        env = StubExecutionEnv(T=args.T, Q0=args.Q0, max_trade_size=args.Q0, seed=args.seed)
        print("[immediate] Using synthetic StubExecutionEnv")
    elif env_name == "tiny":
        env = TinyLOBEnv(
            n_steps=args.T,
            total_inventory=int(args.Q0),
            n_actions=len(DEFAULT_ACTION_SPEC.fractions),
            initial_price=args.initial_price,
            tick_size=args.tick_size,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            spread_mean=args.spread_mean,
            spread_std=args.spread_std,
            base_depth=args.light_base_depth,
            signal_rho=args.light_signal_rho,
            signal_noise=args.light_signal_noise,
            drift_strength=args.light_drift_strength,
            return_noise=args.light_return_noise,
            depth_sensitivity=args.tiny_depth_sensitivity,
            force_liquidation=args.tiny_force_liquidation,
            terminal_impact_mult=args.tiny_terminal_impact_mult,
            seed=args.seed,
        )
        print("[immediate] Using TinyLOBEnv")
    else:
        env = LightLOBEnv(
            n_steps=args.T,
            total_inventory=int(args.Q0),
            n_actions=len(DEFAULT_ACTION_SPEC.fractions),
            initial_price=args.initial_price,
            tick_size=args.tick_size,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            spread_mean=args.spread_mean,
            spread_std=args.spread_std,
            base_depth=args.light_base_depth,
            signal_rho=args.light_signal_rho,
            signal_noise=args.light_signal_noise,
            drift_strength=args.light_drift_strength,
            return_noise=args.light_return_noise,
            participation_impact=args.light_participation_impact,
            seed=args.seed,
        )
        print("[immediate] Using LightLOBEnv")

    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0
        remaining = float(args.Q0)

        while not done:
            target_qty = remaining if step_idx == 0 else 0.0
            if hasattr(env, "step_with_qty"):
                _, reward, terminated, truncated, info = env.step_with_qty(int(round(target_qty)))
                action = -1
            else:
                action = 4 if target_qty > 0 else 0
                _, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            exec_price = float(info.get("exec_price", float("nan")))
            mid_price = float(info.get("mid_price", float("nan")))
            executed_qty = float(info.get("executed_qty", 0.0))
            if executed_qty > 0 and not (
                exec_price != exec_price or mid_price != mid_price
            ):
                step_is = executed_qty * (mid_price - exec_price)
            else:
                step_is = 0.0

            ep_reward += float(reward)
            ep_is += step_is

            remaining = float(info.get("remaining_inventory", info.get("inventory_remaining", 0.0)))
            t_val = int(info.get("t", info.get("step", step_idx)))

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": t_val,
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

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_is": ep_is,
                "completed": int(remaining <= 1e-6),
                "steps": step_idx,
                "epsilon": 0.0,
                "strategy": "immediate",
                "is_twap": 0,
            }
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an immediate-liquidation baseline on an execution env.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run directory before writing.")
    parser.add_argument(
        "--env",
        type=str,
        default="stub",
        choices=["stub", "light", "tiny", "abides"],
        help="Environment backend.",
    )
    parser.add_argument(
        "--use-abides",
        action="store_true",
        help="Backward-compatible alias for --env abides.",
    )
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--spread-mean", type=float, default=0.05)
    parser.add_argument("--spread-std", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=2.5e-6)
    parser.add_argument("--gamma-perm", type=float, default=2.5e-7)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--urgency-coef", type=float, default=0.0)
    parser.add_argument("--light-base-depth", type=float, default=1500.0)
    parser.add_argument("--light-signal-rho", type=float, default=0.85)
    parser.add_argument("--light-signal-noise", type=float, default=0.12)
    parser.add_argument("--light-drift-strength", type=float, default=0.0015)
    parser.add_argument("--light-return-noise", type=float, default=0.0009)
    parser.add_argument("--light-participation-impact", type=float, default=0.04)
    parser.add_argument("--tiny-depth-sensitivity", type=float, default=0.55)
    parser.add_argument("--tiny-terminal-impact-mult", type=float, default=2.5)
    parser.add_argument(
        "--tiny-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_true",
        help="Force terminal liquidation in TinyLOBEnv (default on).",
    )
    parser.add_argument(
        "--tiny-no-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_false",
        help="Disable forced terminal liquidation in TinyLOBEnv.",
    )
    parser.set_defaults(tiny_force_liquidation=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_path = run_immediate(parse_args())
    print(f"Saved immediate baseline artifacts to: {run_path}")
