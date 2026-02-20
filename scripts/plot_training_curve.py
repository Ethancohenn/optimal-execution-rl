from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.metrics import load_episodes, load_trajectories, reward_by_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot episode reward vs episode index.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing episodes.csv.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/figures/training_curve.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Moving-average window for reward smoothing. Set 1 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)

    episodes_df = load_episodes(run_dir)
    if "episode_reward" in episodes_df.columns:
        rewards = episodes_df[["episode", "episode_reward"]].copy()
    else:
        traj_df = load_trajectories(run_dir)
        rewards = reward_by_episode(traj_df).reset_index(name="episode_reward")

    rewards = rewards.sort_values("episode").reset_index(drop=True)
    smooth_window = max(1, int(args.smooth_window))
    rewards["reward_smooth"] = (
        rewards["episode_reward"].rolling(window=smooth_window, min_periods=1).mean()
    )

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(rewards["episode"], rewards["episode_reward"], alpha=0.35, linewidth=1.0, label="Episode reward")
    plt.plot(rewards["episode"], rewards["reward_smooth"], linewidth=2.0, label=f"MA({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training Curve: Episode Reward vs Episode")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
