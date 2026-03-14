"""
Side-by-side strategy comparison plots: RL strategy vs one baseline.

Generates a single figure with 4 panels:
  1. Episode IS boxplot
  2. Episode reward boxplot
  3. Avg exec slippage/step boxplot
  4. Completion rate bar chart

Usage
-----
  conda run -n cs234_rl python scripts/plot_comparison.py \\
      --rl-run-dir runs/dqn_abides_final \\
      --baseline-run-dir runs/immediate_abides_final \\
      --baseline-label Immediate \\
      --eval-last 400 \\
      --out reports/figures/comparison_immediate.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.metrics import load_episodes, load_trajectories

DQN_COLOR = "#3A86FF"
BASELINE_COLOR = "#FF6B6B"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def _tail(df: pd.DataFrame, k: int) -> pd.DataFrame:
    eps = sorted(df["episode"].unique())
    keep = set(eps[-k:])
    return df[df["episode"].isin(keep)]


def _pretty_strategy_name(raw_name: str) -> str:
    mapping = {
        "dqn": "DQN",
        "dqn_eval": "DQN",
        "double_dqn": "Double DQN",
        "double_dqn_eval": "Double DQN",
        "tabular_q": "Tabular Q",
        "tabular_q_eval": "Tabular Q",
        "twap": "TWAP",
        "immediate": "Immediate",
        "last_minute": "Last Minute",
    }
    return mapping.get(raw_name, raw_name.replace("_", " ").title())


def _infer_rl_label(episodes_df: pd.DataFrame, rl_run_dir: str) -> str:
    if "strategy" in episodes_df.columns and not episodes_df["strategy"].dropna().empty:
        return _pretty_strategy_name(str(episodes_df["strategy"].dropna().iloc[0]))

    config_path = Path(rl_run_dir) / "config.json"
    if config_path.exists():
        import json

        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        return _pretty_strategy_name(str(config.get("algorithm", "rl")))

    return "RL"


def _boxplot_panel(
    ax: plt.Axes,
    data_rl: np.ndarray,
    data_baseline: np.ndarray,
    rl_label: str,
    baseline_label: str,
    ylabel: str,
    title: str,
    lower_is_better: bool = True,
) -> None:
    bp = ax.boxplot(
        [data_rl, data_baseline],
        patch_artist=True,
        showmeans=True,
        widths=0.45,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], [DQN_COLOR, BASELINE_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("grey")

    ax.set_xticks([1, 2])
    ax.set_xticklabels([rl_label, baseline_label], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    rl_mean = float(np.mean(data_rl))
    baseline_mean = float(np.mean(data_baseline))
    ax.text(1, ax.get_ylim()[0], f"mu={rl_mean:.1f}", ha="center", va="bottom", fontsize=8, color=DQN_COLOR)
    ax.text(
        2,
        ax.get_ylim()[0],
        f"mu={baseline_mean:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=BASELINE_COLOR,
    )

    if lower_is_better:
        winner = 1 if rl_mean < baseline_mean else 2
    else:
        winner = 1 if rl_mean > baseline_mean else 2
    ax.annotate("better", xy=(winner, ax.get_ylim()[1]), ha="center", va="top", fontsize=8, color="green")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-run-dir", required=True)
    p.add_argument("--baseline-run-dir", default=None)
    p.add_argument("--twap-run-dir", default=None, help="Backward-compatible alias for --baseline-run-dir")
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--rl-label", default=None)
    p.add_argument("--out", default="reports/figures/comparison.png")
    p.add_argument("--eval-last", type=int, default=400, help="Use last N RL episodes as eval set")
    args = p.parse_args()

    baseline_run_dir = args.baseline_run_dir or args.twap_run_dir
    if baseline_run_dir is None:
        raise ValueError("Provide --baseline-run-dir (or legacy --twap-run-dir).")

    rl_ep = load_episodes(args.rl_run_dir)
    baseline_ep = load_episodes(baseline_run_dir)
    rl_traj = load_trajectories(args.rl_run_dir)
    baseline_traj = load_trajectories(baseline_run_dir)
    rl_label = args.rl_label or _infer_rl_label(rl_ep, args.rl_run_dir)

    rl_ep_eval = _tail(rl_ep, args.eval_last)
    rl_traj_eval = _tail(rl_traj, args.eval_last)

    rl_is = rl_ep_eval["episode_is"].values
    baseline_is = baseline_ep["episode_is"].values
    rl_reward = rl_ep_eval["episode_reward"].values
    baseline_reward = baseline_ep["episode_reward"].values

    rl_slip = (rl_traj_eval["exec_price"] - rl_traj_eval["mid_price"]).dropna().values
    baseline_slip = (baseline_traj["exec_price"] - baseline_traj["mid_price"]).dropna().values

    rl_comp = float(rl_ep_eval["completed"].mean()) if "completed" in rl_ep_eval.columns else float("nan")
    baseline_comp = (
        float(baseline_ep["completed"].mean()) if "completed" in baseline_ep.columns else float("nan")
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"{rl_label} vs {args.baseline_label} - Strategy Comparison ({rl_label} eval: last {args.eval_last} episodes)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    _boxplot_panel(
        axes[0],
        rl_is,
        baseline_is,
        rl_label,
        args.baseline_label,
        "Implementation Shortfall",
        "IS per Episode (lower is better)",
        lower_is_better=True,
    )
    _boxplot_panel(
        axes[1],
        rl_reward,
        baseline_reward,
        rl_label,
        args.baseline_label,
        "Episode Reward",
        "Episode Reward (higher is better)",
        lower_is_better=False,
    )
    _boxplot_panel(
        axes[2],
        rl_slip,
        baseline_slip,
        rl_label,
        args.baseline_label,
        "exec - mid price ($)",
        "Execution Slippage / Step (closer to 0 is better)",
        lower_is_better=True,
    )

    ax4 = axes[3]
    bars = ax4.bar(
        [rl_label, args.baseline_label],
        [rl_comp * 100, baseline_comp * 100],
        color=[DQN_COLOR, BASELINE_COLOR],
        alpha=0.8,
        width=0.45,
    )
    for bar, val in zip(bars, [rl_comp, baseline_comp]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1%}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    winner_label = rl_label if rl_comp >= baseline_comp else args.baseline_label
    winner_x = 0.25 if winner_label == rl_label else 0.75
    ax4.text(winner_x, 0.97, "better", ha="center", va="top", transform=ax4.transAxes, color="green", fontsize=9)
    ax4.set_ylabel("Completion Rate (%)", fontsize=10)
    ax4.set_title("Liquidation Completion (higher is better)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0, 115)
    ax4.grid(axis="x", alpha=0)

    fig.tight_layout()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {args.out}")

    print("\nSummary")
    print(f"{'Metric':<30} {rl_label:>12} {args.baseline_label:>12} {'Winner':>12}")
    print("-" * 72)
    rows = [
        ("Mean IS", float(np.mean(rl_is)), float(np.mean(baseline_is)), "lower"),
        ("Mean Reward", float(np.mean(rl_reward)), float(np.mean(baseline_reward)), "higher"),
        ("Mean Slippage", float(np.mean(rl_slip)), float(np.mean(baseline_slip)), "lower_abs"),
        ("Completion Rate", rl_comp, baseline_comp, "higher"),
    ]
    for name, rv, bv, prefer in rows:
        if prefer == "lower":
            winner = rl_label if rv < bv else args.baseline_label
        elif prefer == "higher":
            winner = rl_label if rv > bv else args.baseline_label
        else:
            winner = rl_label if abs(rv) < abs(bv) else args.baseline_label
        print(f"{name:<30} {rv:>12.3f} {bv:>12.3f} {winner:>12}")


if __name__ == "__main__":
    main()
