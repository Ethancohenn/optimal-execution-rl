from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.logger import RunLogger
from src.common.run_dirs import build_run_dir as build_run_dir_common
from src.envs.stub_env import StubExecutionEnv


@dataclass
class QConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200


def discretize_obs(obs: np.ndarray, inv_bins: int = 10, time_bins: int = 10) -> tuple[int, int]:
    inv = float(np.clip(obs[0], 0.0, 1.0))
    time_remaining = float(np.clip(obs[1], 0.0, 1.0))
    inv_idx = min(inv_bins - 1, int(inv * inv_bins))
    time_idx = min(time_bins - 1, int(time_remaining * time_bins))
    return inv_idx, time_idx


def epsilon_for_episode(ep: int, cfg: QConfig) -> float:
    if cfg.epsilon_decay_episodes <= 0:
        return cfg.epsilon_end
    frac = min(1.0, ep / float(cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(
    q_table: np.ndarray,
    state: tuple[int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[-1]))
    return int(np.argmax(q_table[state[0], state[1], :]))


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    return build_run_dir_common(
        base_dir=base_dir,
        run_name=run_name,
        overwrite=overwrite,
        default_prefix="stub_q",
    )


def train(args: argparse.Namespace) -> str:
    env = StubExecutionEnv(T=args.T, Q0=args.Q0, max_trade_size=args.max_trade_size, seed=args.seed)
    run_dir = build_run_dir(args.base_run_dir, args.run_name, overwrite=args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")
    q_cfg = QConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
    )

    rng = np.random.default_rng(args.seed)
    q_table = np.zeros((args.inv_bins, args.time_bins, env.action_space.n), dtype=np.float64)

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        state = discretize_obs(obs, inv_bins=args.inv_bins, time_bins=args.time_bins)
        epsilon = epsilon_for_episode(episode, q_cfg)

        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0

        while not done:
            action = select_action(q_table, state, epsilon, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_obs(next_obs, inv_bins=args.inv_bins, time_bins=args.time_bins)
            done = bool(terminated or truncated)

            td_target = reward + q_cfg.gamma * float(np.max(q_table[next_state[0], next_state[1], :])) * (0.0 if done else 1.0)
            td_error = td_target - q_table[state[0], state[1], action]
            q_table[state[0], state[1], action] += q_cfg.alpha * td_error

            ep_reward += float(reward)
            step_is = float(info.get("step_is", float(info["executed_qty"]) * (float(info["mid_price"]) - float(info["exec_price"]))))
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
                    "is_twap": 0,
                    "executed_qty": float(info["executed_qty"]),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": step_is,
                }
            )

            state = next_state
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": ep_reward,
                "episode_is": ep_is,
                "completed": int(env.remaining_inventory <= 1e-6),
                "steps": step_idx,
                "epsilon": epsilon,
                "strategy": "rl",
                "is_twap": 0,
            }
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple Q-learning agent on StubExecutionEnv.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--max-trade-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inv-bins", type=int, default=10)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=200)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_path = train(parse_args())
    print(f"Saved run artifacts to: {run_path}")
