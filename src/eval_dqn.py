from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.dqn import QNetwork
from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN or Double DQN policy.")
    parser.add_argument("--model-path", required=True, help="Path to policy_net.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--npz-path", default="data/features.npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    return parser.parse_args()


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool, default_prefix: str) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix=default_prefix,
    )


def pretty_algo_name(algo_name: str) -> str:
    mapping = {
        "dqn": "DQN",
        "dqn_eval": "DQN",
        "double_dqn": "Double DQN",
        "double_dqn_eval": "Double DQN",
    }
    return mapping.get(algo_name, algo_name.replace("_", " ").title())


def load_training_metadata(model_path: str) -> dict:
    config_path = Path(model_path).parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def build_env(eval_args: argparse.Namespace, train_meta: dict):
    train_args = train_meta.get("args", {})
    env_name = train_meta.get("environment", "abides_replay" if train_args.get("use_abides") else "light_lob")

    if env_name == "abides_replay":
        norm_stats = None
        norm_path = Path(eval_args.model_path).parent / "norm_stats.json"
        if norm_path.exists():
            with open(norm_path, encoding="utf-8") as handle:
                norm_stats = json.load(handle)

        env = AbidesReplayEnv(
            npz_path=eval_args.npz_path if eval_args.npz_path else train_args.get("npz_path", "data/features.npz"),
            n_steps=int(train_args.get("n_steps", 60)),
            total_inventory=int(train_args.get("total_inventory", 1000)),
            n_actions=int(train_args.get("n_actions", 5)),
            eta=float(train_args.get("eta", 2.5e-6)),
            gamma_perm=float(train_args.get("gamma_perm", 2.5e-7)),
            lambda_penalty=float(train_args.get("lambda_penalty", 1.0)),
            urgency_coef=0.0,
            norm_stats=norm_stats,
            split=eval_args.split,
        )
        return env, env_name

    cfg = EnvConfig(
        total_inventory=int(train_args.get("total_inventory", 1000)),
        n_steps=int(train_args.get("n_steps", 60)),
        n_actions=int(train_args.get("n_actions", 5)),
        initial_price=float(train_args.get("initial_price", 100.0)),
        tick_size=float(train_args.get("tick_size", 0.01)),
        volatility=float(train_args.get("volatility", 0.02)),
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
        fundamental_mean_reversion=float(train_args.get("fundamental_mean_reversion", 0.10)),
        order_flow_memory=float(train_args.get("order_flow_memory", 0.25)),
        agent_permanent_impact=float(train_args.get("agent_permanent_impact", 1.5)),
        lambda_penalty=float(train_args.get("lambda_penalty", 1.0)),
        urgency_penalty=0.0,
        force_liquidation=bool(train_args.get("force_liquidation", True)),
        seed=int(train_args.get("seed", eval_args.seed)),
    )
    return ExecutionEnv(config=cfg), env_name


def main() -> str:
    args = parse_args()
    train_meta = load_training_metadata(args.model_path)
    train_args = train_meta.get("args", {})
    algo_name = str(train_meta.get("algorithm", "dqn"))

    env, env_name = build_env(args, train_meta)
    hidden_dims = train_args.get("hidden_dims", args.hidden_dims)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(state_dim, action_dim, list(hidden_dims)).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    print(f"[eval_dqn] Loaded {pretty_algo_name(algo_name)} model from: {args.model_path}")
    print(f"[eval_dqn] Environment={env_name}")

    def greedy_action(state):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(net(state_t).argmax(dim=1).item())

    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite, f"{algo_name}_eval")
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0

        while not done:
            action = greedy_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
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

            state = next_state
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
                "strategy": f"{algo_name}_eval",
                "is_twap": 0,
            }
        )

    print(f"Saved {pretty_algo_name(algo_name)} eval artifacts to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
