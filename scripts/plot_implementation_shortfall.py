from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.metrics import (
    implementation_shortfall_by_episode,
    load_episodes,
    load_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot implementation shortfall comparison (RL vs baseline).")
    parser.add_argument("--rl-run-dir", type=str, required=True, help="RL run directory.")
    parser.add_argument(
        "--rl-label",
        type=str,
        default="RL",
        help="Display name for the RL run.",
    )
    parser.add_argument("--baseline-run-dir", type=str, default=None, help="Baseline run directory.")
    parser.add_argument(
        "--twap-run-dir",
        type=str,
        default=None,
        help="Backward-compatible alias for --baseline-run-dir.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Display name used in the figure/table.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/figures/implementation_shortfall.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--tail-k",
        type=int,
        default=None,
        help="If set, evaluate only the last K episodes from each run.",
    )
    return parser.parse_args()


def filter_tail(
    episodes_df: pd.DataFrame, traj_df: pd.DataFrame, tail_k: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tail_k is None:
        return episodes_df, traj_df
    tail_k = max(1, int(tail_k))
    keep_eps = episodes_df["episode"].astype(int).sort_values().unique()[-tail_k:]
    episodes_out = episodes_df[episodes_df["episode"].astype(int).isin(keep_eps)].copy()
    traj_out = traj_df[traj_df["episode"].astype(int).isin(keep_eps)].copy()
    return episodes_out, traj_out


def summarize(episodes_df: pd.DataFrame, traj_df: pd.DataFrame, is_per_ep: pd.Series) -> dict[str, float]:
    completion = float(episodes_df["completed"].astype(float).mean()) if "completed" in episodes_df.columns else float("nan")
    forced_share = 0.0
    if "forced_qty" in traj_df.columns and "executed_qty" in traj_df.columns:
        total_forced = float(traj_df["forced_qty"].astype(float).sum())
        total_exec = float(traj_df["executed_qty"].astype(float).sum())
        forced_share = (total_forced / total_exec) if total_exec > 0 else 0.0
    return {
        "mean_is": float(is_per_ep.mean()),
        "p95_is": float(is_per_ep.quantile(0.95)),
        "completion_rate": completion,
        "forced_liq_share": forced_share,
    }


def main() -> None:
    args = parse_args()
    baseline_run_dir = args.baseline_run_dir or args.twap_run_dir
    if baseline_run_dir is None:
        raise ValueError("Provide --baseline-run-dir (or legacy --twap-run-dir).")

    rl_eps = load_episodes(args.rl_run_dir)
    baseline_eps = load_episodes(baseline_run_dir)
    rl_traj = load_trajectories(args.rl_run_dir)
    baseline_traj = load_trajectories(baseline_run_dir)

    rl_eps, rl_traj = filter_tail(rl_eps, rl_traj, args.tail_k)
    baseline_eps, baseline_traj = filter_tail(baseline_eps, baseline_traj, args.tail_k)

    rl_is = implementation_shortfall_by_episode(rl_traj)
    baseline_is = implementation_shortfall_by_episode(baseline_traj)

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot([rl_is.values, baseline_is.values], tick_labels=[args.rl_label, args.baseline_label], showmeans=True)
    plt.ylabel("Implementation shortfall")
    plt.title(f"Implementation Shortfall: {args.rl_label} vs {args.baseline_label}")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")

    rl_summary = summarize(rl_eps, rl_traj, rl_is)
    baseline_summary = summarize(baseline_eps, baseline_traj, baseline_is)
    summary = pd.DataFrame(
        {
            "strategy": [args.rl_label, args.baseline_label],
            "mean_is": [rl_summary["mean_is"], baseline_summary["mean_is"]],
            "p95_is": [rl_summary["p95_is"], baseline_summary["p95_is"]],
            "completion_rate": [rl_summary["completion_rate"], baseline_summary["completion_rate"]],
            "forced_liq_share": [
                rl_summary.get("forced_liq_share", 0.0),
                baseline_summary.get("forced_liq_share", 0.0),
            ],
        }
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
