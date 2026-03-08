"""
Side-by-side strategy comparison plots: DQN vs TWAP.

Generates a single figure with 4 panels:
  1. Episode IS boxplot        (lower = better)
  2. Episode reward boxplot    (higher = better)
  3. Avg exec slippage/step    (exec_price - mid_price per step, lower magnitude = better)
  4. Completion rate bar       (higher = better)

Usage
-----
  conda run -n cs234_rl python scripts/plot_comparison.py \\
      --rl-run-dir   runs/dqn_abides_final \\
      --twap-run-dir runs/twap_abides_final \\
      --eval-last    400 \\
      --out          reports/figures/comparison.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.metrics import load_episodes, load_trajectories

DQN_COLOR  = "#3A86FF"
TWAP_COLOR = "#FF6B6B"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _tail(df: pd.DataFrame, k: int) -> pd.DataFrame:
    eps = sorted(df["episode"].unique())
    keep = set(eps[-k:])
    return df[df["episode"].isin(keep)]


def _boxplot_panel(ax, data_dqn, data_twap, ylabel, title, lower_is_better=True):
    bp = ax.boxplot(
        [data_dqn, data_twap],
        patch_artist=True,
        showmeans=True,
        widths=0.45,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], [DQN_COLOR, TWAP_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("grey")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["DQN", "TWAP"], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate means
    dqn_mean = np.mean(data_dqn)
    twap_mean = np.mean(data_twap)
    ax.text(1, ax.get_ylim()[0], f"μ={dqn_mean:.1f}", ha="center", va="bottom",
            fontsize=8, color=DQN_COLOR)
    ax.text(2, ax.get_ylim()[0], f"μ={twap_mean:.1f}", ha="center", va="bottom",
            fontsize=8, color=TWAP_COLOR)

    # Winner arrow
    if lower_is_better:
        winner = 1 if dqn_mean < twap_mean else 2
    else:
        winner = 1 if dqn_mean > twap_mean else 2
    ax.annotate("★ better", xy=(winner, ax.get_ylim()[1]),
                ha="center", va="top", fontsize=8, color="green", fontweight="bold")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-run-dir",   required=True)
    p.add_argument("--twap-run-dir", required=True)
    p.add_argument("--out",      default="reports/figures/comparison.png")
    p.add_argument("--eval-last", type=int, default=400,
                   help="Use last N DQN episodes as eval set")
    args = p.parse_args()

    dqn_ep    = load_episodes(args.rl_run_dir)
    twap_ep   = load_episodes(args.twap_run_dir)
    dqn_traj  = load_trajectories(args.rl_run_dir)
    twap_traj = load_trajectories(args.twap_run_dir)

    # Filter DQN to eval episodes
    dqn_ep_eval   = _tail(dqn_ep,   args.eval_last)
    dqn_traj_eval = _tail(dqn_traj, args.eval_last)

    # ── Panel data ────────────────────────────────────────────────
    dqn_is     = dqn_ep_eval["episode_is"].values
    twap_is    = twap_ep["episode_is"].values
    dqn_reward = dqn_ep_eval["episode_reward"].values
    twap_reward= twap_ep["episode_reward"].values

    # Avg step slippage: exec_price - mid_price  (selling → want exec high → close to mid)
    dqn_slip  = (dqn_traj_eval["exec_price"]  - dqn_traj_eval["mid_price"]).dropna().values
    twap_slip = (twap_traj["exec_price"] - twap_traj["mid_price"]).dropna().values

    # Completion rate
    dqn_comp  = float(dqn_ep_eval["completed"].mean())  if "completed" in dqn_ep_eval.columns  else float("nan")
    twap_comp = float(twap_ep["completed"].mean())       if "completed" in twap_ep.columns       else float("nan")

    # ── Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"DQN vs TWAP — Strategy Comparison  (DQN eval: last {args.eval_last} episodes)",
                 fontsize=13, fontweight="bold", y=1.01)

    # 1. IS
    _boxplot_panel(axes[0], dqn_is, twap_is,
                   "Implementation Shortfall", "IS per Episode\n(lower = better)",
                   lower_is_better=True)

    # 2. Reward
    _boxplot_panel(axes[1], dqn_reward, twap_reward,
                   "Episode Reward", "Episode Reward\n(higher = better)",
                   lower_is_better=False)

    # 3. Slippage per step
    _boxplot_panel(axes[2], dqn_slip, twap_slip,
                   "exec − mid price ($)", "Execution Slippage / Step\n(closer to 0 = better)",
                   lower_is_better=True)

    # 4. Completion rate (bar)
    ax4 = axes[3]
    bars = ax4.bar(["DQN", "TWAP"], [dqn_comp * 100, twap_comp * 100],
                   color=[DQN_COLOR, TWAP_COLOR], alpha=0.8, width=0.45)
    for bar, val in zip(bars, [dqn_comp, twap_comp]):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    winner_label = "DQN" if dqn_comp >= twap_comp else "TWAP"
    winner_x = 0.25 if winner_label == "DQN" else 0.75
    ax4.text(winner_x, 0.97, "★ better",
             ha="center", va="top", transform=ax4.transAxes,
             color="green", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Completion Rate (%)", fontsize=10)
    ax4.set_title("Liquidation Completion\n(higher = better)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.grid(axis="x", alpha=0)

    fig.tight_layout()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────")
    print(f"{'Metric':<30} {'DQN':>12} {'TWAP':>12} {'Winner':>8}")
    print("-" * 65)
    rows = [
        ("Mean IS",         np.mean(dqn_is),     np.mean(twap_is),    "lower"),
        ("Mean Reward",     np.mean(dqn_reward),  np.mean(twap_reward), "higher"),
        ("Mean Slippage",   np.mean(dqn_slip),    np.mean(twap_slip),  "lower|abs"),
        ("Completion Rate", dqn_comp,             twap_comp,           "higher"),
    ]
    for name, dv, tv, prefer in rows:
        if prefer == "lower":
            winner = "DQN ✓" if dv < tv else "TWAP ✓"
        elif prefer == "higher":
            winner = "DQN ✓" if dv > tv else "TWAP ✓"
        else:
            winner = "DQN ✓" if abs(dv) < abs(tv) else "TWAP ✓"
        print(f"{name:<30} {dv:>12.3f} {tv:>12.3f} {winner:>8}")


if __name__ == "__main__":
    main()
