from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.dqn import QNetwork
from src.common.logger import RunLogger


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("dqn2_eval_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise ValueError(f"Run dir exists: {run_dir}. Use --overwrite.")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved DQN2 policy with IS-aligned reward accounting.")
    p.add_argument("--model-path", required=True, help="Path to policy_net.pt")
    p.add_argument("--npz-path", default="data/features.npz")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--total-inventory", type=int, default=1000)
    p.add_argument("--n-actions", type=int, default=5)
    p.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--leftover-penalty-coef", type=float, default=0.01)
    p.add_argument(
        "--hold-penalty-coef",
        type=float,
        default=0.01,
        help=(
            "Dense per-step inventory holding penalty coefficient. "
            "Applied as: reward -= coef * (inventory_remaining / total_inventory) * benchmark_price."
        ),
    )
    p.add_argument("--env-urgency-coef", type=float, default=0.0)
    p.add_argument("--env-lambda-penalty", type=float, default=0.0)
    p.add_argument("--base-run-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def compute_step_is(info: dict) -> float:
    executed_qty = float(info.get("executed_qty", 0.0))
    if executed_qty <= 0:
        return 0.0

    exec_price = float(info.get("exec_price", float("nan")))
    mid_price = float(info.get("mid_price", float("nan")))
    if np.isnan(exec_price) or np.isnan(mid_price):
        return 0.0

    return executed_qty * (mid_price - exec_price)


def objective_reward_from_info(
    info: dict,
    done: bool,
    leftover_penalty_coef: float,
    hold_penalty_coef: float,
    total_inventory: float,
) -> tuple[float, float, float, float]:
    step_is = compute_step_is(info)
    hold_penalty = 0.0
    leftover_penalty = 0.0

    if hold_penalty_coef > 0.0:
        inv_left = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        benchmark_price = float(info.get("benchmark_price", 1.0))
        inv_frac = inv_left / max(float(total_inventory), 1.0)
        hold_penalty = float(hold_penalty_coef * inv_frac * max(benchmark_price, 1.0))

    if done and leftover_penalty_coef > 0.0:
        leftover = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        benchmark_price = float(info.get("benchmark_price", 1.0))
        if leftover > 0.0:
            leftover_penalty = float(leftover_penalty_coef * leftover * max(benchmark_price, 1.0))

    reward = -step_is - hold_penalty - leftover_penalty
    return float(reward), float(step_is), float(hold_penalty), float(leftover_penalty)


def main() -> str:
    args = parse_args()

    norm_stats = None
    norm_path = Path(args.model_path).parent / "norm_stats.json"
    if norm_path.exists():
        with open(norm_path, encoding="utf-8") as f:
            norm_stats = json.load(f)
        print(f"[eval_dqn2] Loaded norm_stats from: {norm_path}")
    else:
        print("[eval_dqn2] Warning: norm_stats.json not found, using eval-data stats.")

    env = AbidesReplayEnv(
        npz_path=args.npz_path,
        n_steps=args.n_steps,
        total_inventory=args.total_inventory,
        n_actions=args.n_actions,
        norm_stats=norm_stats,
        split=args.split,
        urgency_coef=args.env_urgency_coef,
        lambda_penalty=args.env_lambda_penalty,
    )
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(state_dim, action_dim, args.hidden_dims).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    print(f"[eval_dqn2] Loaded model from: {args.model_path}")
    print(f"[eval_dqn2] Test data: {args.npz_path} (split={args.split})")
    if args.leftover_penalty_coef <= 0.0 and args.hold_penalty_coef <= 0.0:
        print(
            "[eval_dqn2] Warning: leftover_penalty_coef <= 0 and hold_penalty_coef <= 0, "
            "episode_reward may not penalize non-completion/waiting."
        )
    print(
        f"[eval_dqn2] Objective=is_aligned, "
        f"hold_penalty_coef={args.hold_penalty_coef}, "
        f"leftover_penalty_coef={args.leftover_penalty_coef}"
    )

    def greedy_action(state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(net(s).argmax(dim=1).item())

    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite)
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")
    total_exec_qty = 0.0
    completed_count = 0

    for episode in range(args.episodes):
        state, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_env_reward = 0.0
        ep_is = 0.0
        ep_hold_penalty = 0.0
        ep_leftover_penalty = 0.0
        step_idx = 0

        while not done:
            action = greedy_action(state)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            reward, step_is, hold_penalty, leftover_penalty = objective_reward_from_info(
                info=info,
                done=done,
                leftover_penalty_coef=args.leftover_penalty_coef,
                hold_penalty_coef=args.hold_penalty_coef,
                total_inventory=float(args.total_inventory),
            )

            executed_qty = float(info.get("executed_qty", 0.0))
            exec_price = float(info.get("exec_price", float("nan")))
            mid_price = float(info.get("mid_price", float("nan")))
            t_val = int(info.get("t", info.get("step", step_idx)))
            remaining = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))

            ep_reward += float(reward)
            ep_env_reward += float(env_reward)
            ep_is += float(step_is)
            ep_hold_penalty += float(hold_penalty)
            ep_leftover_penalty += float(leftover_penalty)
            total_exec_qty += executed_qty

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
                    "env_reward": float(env_reward),
                    "is_twap": 0,
                    "executed_qty": executed_qty,
                    "forced_qty": 0.0,
                    "step_is": float(step_is),
                    "hold_penalty": float(hold_penalty),
                    "leftover_penalty": float(leftover_penalty),
                }
            )

            state = next_state
            step_idx += 1

        remaining = float(info.get("inventory_remaining", 0.0))
        completed = int(remaining <= 1e-6)
        completed_count += completed
        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_env_reward": float(ep_env_reward),
                "episode_is": float(ep_is),
                "episode_hold_penalty": float(ep_hold_penalty),
                "episode_leftover_penalty": float(ep_leftover_penalty),
                "completed": completed,
                "mean_loss": 0.0,
                "epsilon": 0.0,
                "steps": step_idx,
                "strategy": "dqn2_eval_is_aligned",
                "is_twap": 0,
            }
        )

    print(f"Saved DQN2 eval artifacts to: {run_dir}")
    completion_rate = completed_count / max(1, int(args.episodes))
    print(f"[eval_dqn2] completion_rate={completion_rate:.1%} total_executed_qty={total_exec_qty:.0f}")
    if total_exec_qty <= 0.0:
        print(
            "[eval_dqn2] Warning: policy executed zero shares across evaluation. "
            "Increase leftover penalty strength or retrain."
        )
    return run_dir


if __name__ == "__main__":
    main()
