"""
Evaluate a saved DQN policy (policy_net.pt) on a held-out test environment.
No training — pure greedy inference (epsilon=0).

Usage
-----
  conda run -n cs234_rl python -m src.eval_dqn \\
      --model-path runs/dqn_abides_final/policy_net.pt \\
      --npz-path   data/features.npz \\
      --episodes   200 \\
      --run-name   dqn_test_eval

Output
------
  runs/dqn_test_eval/episodes.csv
  runs/dqn_test_eval/trajectories.csv
"""
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

from src.agents.dqn import QNetwork
from src.common.actions import DEFAULT_ACTION_SPEC
from src.common.logger import RunLogger
from execution_infra.abides_replay_env import AbidesReplayEnv


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool) -> str:
    if run_name is None:
        run_name = datetime.now().strftime("dqn_eval_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise ValueError(f"Run dir exists: {run_dir}. Use --overwrite.")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved DQN policy on test data.")
    p.add_argument("--model-path",  required=True, help="Path to policy_net.pt")
    p.add_argument("--npz-path",    default="data/features_test.npz")
    p.add_argument("--episodes",    type=int,   default=200)
    p.add_argument("--n-steps",     type=int,   default=60)
    p.add_argument("--total-inventory", type=int, default=1000)
    p.add_argument("--n-actions",   type=int,   default=5)
    p.add_argument("--split",       type=str,   default="test", choices=["train", "test", "all"])
    p.add_argument("--hidden-dims", type=int,   nargs="+", default=[128, 128])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--base-run-dir",type=str,   default="runs")
    p.add_argument("--run-name",    type=str,   default=None)
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load normalisation stats (so test obs matches training distribution) ─
    norm_stats = None
    norm_path = Path(args.model_path).parent / "norm_stats.json"
    if norm_path.exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        print(f"[eval_dqn] Loaded norm_stats from: {norm_path}")
    else:
        print("[eval_dqn] Warning: norm_stats.json not found — using test-data stats (may be OOD)")

    # ── Environment ───────────────────────────────────────────────
    env = AbidesReplayEnv(
        npz_path=args.npz_path,
        n_steps=args.n_steps,
        total_inventory=args.total_inventory,
        n_actions=args.n_actions,
        norm_stats=norm_stats,
        split=args.split,
        urgency_coef=0.0,
        lambda_penalty=1.0,
    )
    state_dim  = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    # ── Load trained policy (greedy, no training) ─────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(state_dim, action_dim, args.hidden_dims).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    print(f"[eval_dqn] Loaded model from: {args.model_path}")
    print(f"[eval_dqn] Test data: {args.npz_path}")

    def greedy_action(state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(net(s).argmax(dim=1).item())

    # ── Run eval episodes ─────────────────────────────────────────
    run_dir = build_run_dir(args.base_run_dir, args.run_name, args.overwrite)
    logger  = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    for episode in range(args.episodes):
        state, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is     = 0.0
        step_idx  = 0

        while not done:
            action = greedy_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            executed_qty = float(info.get("executed_qty", 0))
            exec_price   = float(info.get("exec_price",   float("nan")))
            mid_price    = float(info.get("mid_price",    float("nan")))
            step_is = (
                executed_qty * (mid_price - exec_price)
                if (executed_qty > 0 and not (np.isnan(exec_price) or np.isnan(mid_price)))
                else 0.0
            )

            ep_reward += float(reward)
            ep_is     += step_is

            logger.log_step({
                "episode":              episode,
                "step":                 step_idx,
                "t":                    int(info["step"]),
                "remaining_inventory":  float(info["inventory_remaining"]),
                "action":               int(action),
                "exec_price":           exec_price,
                "mid_price":            mid_price,
                "reward":               float(reward),
                "is_twap":              0,
                "executed_qty":         executed_qty,
                "forced_qty":           0.0,
                "step_is":              step_is,
            })
            state    = next_state
            step_idx += 1

        remaining = float(info.get("inventory_remaining", 0.0))
        logger.log_episode({
            "episode":        episode,
            "episode_reward": ep_reward,
            "episode_is":     ep_is,
            "completed":      int(remaining <= 1e-6),
            "mean_loss":      0.0,
            "epsilon":        0.0,
            "steps":          step_idx,
            "strategy":       "dqn_eval",
            "is_twap":        0,
        })

    print(f"Saved DQN eval artifacts to: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
