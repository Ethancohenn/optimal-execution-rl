from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.tabular_q import DiscretizerSpec, ObservationDiscretizer, TabularQAgent
from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on the execution environment.")

    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--total-inventory", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--spread-mean", type=float, default=0.05)
    parser.add_argument("--spread-std", type=float, default=0.01)
    parser.add_argument("--lob-depth-mean", type=int, default=500)
    parser.add_argument("--lob-depth-std", type=int, default=100)
    parser.add_argument("--n-lob-levels", type=int, default=5)
    parser.add_argument("--depth-decay", type=float, default=0.35)
    parser.add_argument("--obs-depth-levels", type=int, default=3)
    parser.add_argument("--market-lot-size", type=int, default=25)
    parser.add_argument("--market-order-intensity", type=float, default=2.0)
    parser.add_argument("--limit-order-intensity", type=float, default=1.2)
    parser.add_argument("--cancellation-rate", type=float, default=0.04)
    parser.add_argument("--quote-improve-prob", type=float, default=0.15)
    parser.add_argument("--volatility", type=float, default=0.02)
    parser.add_argument("--fundamental-mean-reversion", type=float, default=0.10)
    parser.add_argument("--order-flow-memory", type=float, default=0.25)
    parser.add_argument("--agent-permanent-impact", type=float, default=1.5)
    parser.add_argument("--force-liquidation", action="store_true", default=True)
    parser.add_argument("--no-force-liquidation", dest="force_liquidation", action="store_false")

    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--state-bins", nargs=6, type=int, default=list(DiscretizerSpec().bins))

    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--use-abides", action="store_true")
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    return parser.parse_args()


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix="tabular_exec",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_env(args: argparse.Namespace):
    if args.use_abides:
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.n_steps,
            total_inventory=args.total_inventory,
            n_actions=args.n_actions,
            split=args.split,
        )
        env_name = "abides_replay"
    else:
        cfg = EnvConfig(
            total_inventory=args.total_inventory,
            n_steps=args.n_steps,
            n_actions=args.n_actions,
            initial_price=args.initial_price,
            tick_size=args.tick_size,
            spread_mean=args.spread_mean,
            spread_std=args.spread_std,
            lob_depth_mean=args.lob_depth_mean,
            lob_depth_std=args.lob_depth_std,
            n_lob_levels=args.n_lob_levels,
            depth_decay=args.depth_decay,
            obs_depth_levels=args.obs_depth_levels,
            market_lot_size=args.market_lot_size,
            market_order_intensity=args.market_order_intensity,
            limit_order_intensity=args.limit_order_intensity,
            cancellation_rate=args.cancellation_rate,
            quote_improve_prob=args.quote_improve_prob,
            volatility=args.volatility,
            fundamental_mean_reversion=args.fundamental_mean_reversion,
            order_flow_memory=args.order_flow_memory,
            agent_permanent_impact=args.agent_permanent_impact,
            force_liquidation=args.force_liquidation,
            seed=args.seed,
        )
        env = ExecutionEnv(config=cfg)
        env_name = "light_lob"
    return env, env_name


def train(args: argparse.Namespace) -> str:
    set_global_seed(args.seed)
    env, env_name = build_env(args)

    discretizer = ObservationDiscretizer(
        DiscretizerSpec(bins=tuple(args.state_bins))
    )
    agent = TabularQAgent(
        state_bins=tuple(args.state_bins),
        action_dim=int(env.action_space.n),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        seed=args.seed,
    )

    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    with open(os.path.join(run_dir, "tabular_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "env": env_name,
                "state_bins": list(args.state_bins),
                "discretizer": discretizer.metadata(),
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        state = discretizer.transform(obs)

        done = False
        ep_reward = 0.0
        ep_is = 0.0
        ep_td_error = 0.0
        step_idx = 0

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretizer.transform(next_obs)
            done = bool(terminated or truncated)

            td_error = agent.update(state, action, reward, next_state, done)
            ep_td_error += abs(td_error)
            ep_reward += float(reward)
            ep_is += float(info.get("step_is", 0.0))

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
                    "step_is": float(info.get("step_is", 0.0)),
                }
            )

            state = next_state
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(float(info.get("inventory_remaining", 1.0)) <= 1e-6),
                "mean_loss": float(ep_td_error / max(1, step_idx)),
                "epsilon": float(agent.epsilon()),
                "steps": step_idx,
                "strategy": "tabular_q",
                "is_twap": 0,
            }
        )

    agent.save(os.path.join(run_dir, "q_table.npy"))
    return run_dir


if __name__ == "__main__":
    run_path = train(parse_args())
    print(f"Saved tabular Q-learning artifacts to: {run_path}")
