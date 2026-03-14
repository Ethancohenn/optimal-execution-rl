from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.tabular_q import DiscretizerSpec, ObservationDiscretizer, sample_greedy_action
from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved tabular Q-learning policy.")
    parser.add_argument("--model-path", required=True, help="Path to q_table.npy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    return parser.parse_args()


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix="tabular_q_eval",
    )


def load_training_metadata(model_path: str) -> dict:
    meta_path = Path(model_path).parent / "tabular_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_discretizer_spec(train_meta: dict) -> DiscretizerSpec:
    default_spec = DiscretizerSpec()
    disc_meta = train_meta.get("discretizer")
    if disc_meta:
        return DiscretizerSpec.from_dict(disc_meta)

    train_args = train_meta.get("args", {})
    state_bins = train_meta.get("state_bins", train_args.get("state_bins", list(default_spec.bins)))
    return DiscretizerSpec(
        bins=tuple(int(x) for x in state_bins),
        lows=default_spec.lows,
        highs=default_spec.highs,
    )


def build_env(eval_args: argparse.Namespace, train_meta: dict):
    train_args = train_meta.get("args", {})
    env_name = train_meta.get("env", "abides_replay" if train_args.get("use_abides") else "light_lob")

    if env_name == "abides_replay":
        env = AbidesReplayEnv(
            npz_path=eval_args.npz_path if eval_args.npz_path else train_args.get("npz_path", "data/features.npz"),
            n_steps=int(train_args.get("n_steps", 60)),
            total_inventory=int(train_args.get("total_inventory", 1000)),
            n_actions=int(train_args.get("n_actions", 5)),
            split=eval_args.split,
        )
        return env, env_name

    cfg = EnvConfig(
        total_inventory=int(train_args.get("total_inventory", 1000)),
        n_steps=int(train_args.get("n_steps", 60)),
        n_actions=int(train_args.get("n_actions", 5)),
        initial_price=float(train_args.get("initial_price", 100.0)),
        tick_size=float(train_args.get("tick_size", 0.01)),
        spread_mean=float(train_args.get("spread_mean", 0.05)),
        spread_std=float(train_args.get("spread_std", 0.01)),
        lob_depth_mean=int(train_args.get("lob_depth_mean", 500)),
        lob_depth_std=int(train_args.get("lob_depth_std", 100)),
        n_lob_levels=int(train_args.get("n_lob_levels", 5)),
        depth_decay=float(train_args.get("depth_decay", 0.35)),
        obs_depth_levels=int(train_args.get("obs_depth_levels", 3)),
        market_lot_size=int(train_args.get("market_lot_size", 25)),
        market_order_intensity=float(train_args.get("market_order_intensity", 2.0)),
        limit_order_intensity=float(train_args.get("limit_order_intensity", 1.2)),
        cancellation_rate=float(train_args.get("cancellation_rate", 0.04)),
        quote_improve_prob=float(train_args.get("quote_improve_prob", 0.15)),
        volatility=float(train_args.get("volatility", 0.02)),
        fundamental_mean_reversion=float(train_args.get("fundamental_mean_reversion", 0.10)),
        order_flow_memory=float(train_args.get("order_flow_memory", 0.25)),
        agent_permanent_impact=float(train_args.get("agent_permanent_impact", 1.5)),
        force_liquidation=bool(train_args.get("force_liquidation", True)),
        seed=int(train_args.get("seed", eval_args.seed)),
    )
    return ExecutionEnv(config=cfg), env_name


def main() -> str:
    args = parse_args()
    train_meta = load_training_metadata(args.model_path)
    discretizer_spec = load_discretizer_spec(train_meta)
    env, env_name = build_env(args, train_meta)
    discretizer = ObservationDiscretizer(discretizer_spec)
    q_table = np.load(args.model_path)
    rng = np.random.default_rng(args.seed)

    print(f"[eval_tabular] Loaded Tabular Q model from: {args.model_path}")
    print(f"[eval_tabular] Environment={env_name}")

    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        state = discretizer.transform(obs)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0

        while not done:
            action = sample_greedy_action(q_table[state], rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            step_is = float(info.get("step_is", 0.0))
            ep_reward += float(reward)
            ep_is += step_is

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": int(info.get("step", step_idx)),
                    "timestamp_ns": int(info.get("timestamp_ns", -1)),
                    "remaining_inventory": float(info.get("inventory_remaining", 0.0)),
                    "action": int(action),
                    "exec_price": float(info.get("exec_price", float("nan"))),
                    "mid_price": float(info.get("mid_price", float("nan"))),
                    "reward": float(reward),
                    "is_twap": 0,
                    "executed_qty": float(info.get("executed_qty", 0.0)),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": step_is,
                }
            )

            state = discretizer.transform(next_obs)
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(float(info.get("inventory_remaining", 1.0)) <= 1e-6),
                "mean_loss": 0.0,
                "epsilon": 0.0,
                "steps": step_idx,
                "strategy": "tabular_q_eval",
                "is_twap": 0,
            }
        )

    print(f"Saved Tabular Q eval artifacts to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
