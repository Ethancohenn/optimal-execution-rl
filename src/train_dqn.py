from __future__ import annotations

import argparse
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
from src.agents.dqn import DQNAgent
from src.common.logger import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN/Double-DQN on execution_infra.ExecutionEnv.")

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--total-inventory", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--volatility", type=float, default=0.002)
    parser.add_argument("--spread-mean", type=float, default=0.05)
    parser.add_argument("--spread-std", type=float, default=0.01)
    parser.add_argument("--lob-depth-mean", type=int, default=500)
    parser.add_argument("--lob-depth-std", type=int, default=100)
    parser.add_argument("--n-lob-levels", type=int, default=3)
    parser.add_argument("--eta", type=float, default=2.5e-6)
    parser.add_argument("--gamma-perm", type=float, default=2.5e-7)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)

    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--double-dqn", action="store_true", help="Enable Double DQN target computation (default off).")
    parser.add_argument("--smoothness-coef", type=float, default=0.0)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)

    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing run directory before writing.")
    parser.add_argument("--save-model", action="store_true", help="Save policy network to run_dir/policy_net.pt")
    return parser.parse_args()


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("dqn_exec_%Y%m%d_%H%M%S")
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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> str:
    set_global_seed(args.seed)

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
        eta=args.eta,
        gamma_perm=args.gamma_perm,
        lambda_penalty=args.lambda_penalty,
        seed=args.seed,
    )
    env = ExecutionEnv(config=cfg)
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
        double_dqn=args.double_dqn,
        smoothness_coef=args.smoothness_coef,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
    )
    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

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
            inventory_before = int(env.inventory)
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.store_transition(state, action, reward, next_state, done, prev_action)
            metrics = agent.train_step()
            if metrics is not None:
                ep_loss += metrics["loss"]
                loss_count += 1

            frac = env.cfg.action_fractions[action]
            executed_qty = min(int(np.round(frac * inventory_before)), inventory_before)
            step_is = -float(reward)
            ep_is += step_is

            exec_price = float("nan")
            if executed_qty > 0 and env.trades and int(env.trades[-1]["step"]) == int(info["step"]):
                exec_price = float(env.trades[-1]["exec_price"])

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": int(info["step"]),
                    "remaining_inventory": float(info["inventory_remaining"]),
                    "action": int(action),
                    "exec_price": exec_price,
                    "mid_price": float(env.market.mid_price),
                    "reward": float(reward),
                    "is_twap": 0,
                    "executed_qty": float(executed_qty),
                    "forced_qty": 0.0,
                    "step_is": float(step_is),
                }
            )

            ep_reward += reward
            prev_action = int(action)
            state = next_state
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(info["inventory_remaining"] <= 0),
                "mean_loss": float(ep_loss / max(1, loss_count)),
                "epsilon": float(agent.epsilon()),
                "steps": step_idx,
                "strategy": "dqn",
                "is_twap": 0,
            }
        )
    if args.save_model:
        model_path = os.path.join(run_dir, "policy_net.pt")
        torch.save(agent.policy_net.state_dict(), model_path)

    return run_dir


if __name__ == "__main__":
    run_path = train(parse_args())
    print(f"Saved DQN artifacts to: {run_path}")
