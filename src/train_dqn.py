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
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.dqn import DQNAgent
from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN/Double DQN on the execution environment.")

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--total-inventory", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--volatility", type=float, default=0.02)
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
    parser.add_argument("--fundamental-mean-reversion", type=float, default=0.10)
    parser.add_argument("--order-flow-memory", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=2.5e-6)
    parser.add_argument("--gamma-perm", type=float, default=2.5e-7)
    parser.add_argument("--agent-permanent-impact", type=float, default=1.5)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--urgency-coef", type=float, default=2.0)
    parser.add_argument("--force-liquidation", action="store_true", default=True)
    parser.add_argument("--no-force-liquidation", dest="force_liquidation", action="store_false")

    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2_500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "double_dqn"],
        help="Training algorithm. Use 'double_dqn' for Double DQN targets.",
    )
    parser.add_argument("--double-dqn", action="store_true", help="Alias for --algorithm double_dqn.")
    parser.add_argument("--smoothness-coef", type=float, default=0.0)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)

    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--use-abides", action="store_true")
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    return parser.parse_args()


def resolve_algorithm(args: argparse.Namespace) -> tuple[str, bool]:
    use_double_dqn = bool(args.double_dqn or args.algorithm == "double_dqn")
    algo_name = "double_dqn" if use_double_dqn else "dqn"
    return algo_name, use_double_dqn


def build_run_dir(
    base_dir: str,
    run_name: str | None,
    overwrite: bool,
    default_prefix: str,
) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix=default_prefix,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(args: argparse.Namespace):
    if args.use_abides:
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.n_steps,
            total_inventory=args.total_inventory,
            n_actions=args.n_actions,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            split=args.split,
        )
        norm_stats = {"mean_spread": env._mean_spread, "mean_bid_vol": env._mean_bid_vol}
        return env, "abides_replay", norm_stats

    cfg = EnvConfig(
        total_inventory=args.total_inventory,
        n_steps=args.n_steps,
        n_actions=args.n_actions,
        initial_price=args.initial_price,
        tick_size=args.tick_size,
        volatility=args.volatility,
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
        fundamental_mean_reversion=args.fundamental_mean_reversion,
        order_flow_memory=args.order_flow_memory,
        eta=args.eta,
        gamma_perm=args.gamma_perm,
        agent_permanent_impact=args.agent_permanent_impact,
        lambda_penalty=args.lambda_penalty,
        urgency_penalty=args.urgency_coef,
        force_liquidation=args.force_liquidation,
        seed=args.seed,
    )
    return ExecutionEnv(config=cfg), "light_lob", None


def train(args: argparse.Namespace) -> str:
    set_global_seed(args.seed)
    algo_name, use_double_dqn = resolve_algorithm(args)
    env, env_name, norm_stats = build_env(args)

    print(f"[train_dqn] Algorithm={algo_name} (double_dqn={use_double_dqn})")
    print(f"[train_dqn] Environment={env_name}")

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        double_dqn=use_double_dqn,
        smoothness_coef=args.smoothness_coef,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
    )

    run_prefix = "ddqn_exec" if use_double_dqn else "dqn_exec"
    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite, run_prefix)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "algorithm": algo_name,
                "environment": env_name,
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    for episode in trange(args.episodes, desc="train"):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        ep_loss = 0.0
        loss_count = 0
        prev_action = 0
        step_idx = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.store_transition(state, action, reward, next_state, done, prev_action)
            metrics = agent.train_step()
            if metrics is not None:
                ep_loss += metrics["loss"]
                loss_count += 1

            executed_qty = int(info.get("executed_qty", 0))
            step_is = float(info.get("step_is", -reward))
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
                    "executed_qty": float(executed_qty),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": step_is,
                }
            )

            ep_reward += float(reward)
            prev_action = int(action)
            state = next_state
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(float(info.get("inventory_remaining", 1.0)) <= 1e-6),
                "mean_loss": float(ep_loss / max(1, loss_count)),
                "epsilon": float(agent.epsilon()),
                "steps": step_idx,
                "strategy": algo_name,
                "is_twap": 0,
            }
        )

    if args.save_model:
        torch.save(agent.policy_net.state_dict(), os.path.join(run_dir, "policy_net.pt"))
        if norm_stats is not None:
            with open(os.path.join(run_dir, "norm_stats.json"), "w", encoding="utf-8") as handle:
                json.dump(norm_stats, handle, indent=2)

    return run_dir


if __name__ == "__main__":
    args = parse_args()
    algo_name, _ = resolve_algorithm(args)
    run_path = train(args)
    print(f"Saved {algo_name} artifacts to: {run_path}")
