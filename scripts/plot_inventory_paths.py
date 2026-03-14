"""
Plot mean inventory depletion paths for RL and a baseline on the same axes.
Supports either step index (default) or elapsed wall-clock time from
``timestamp_ns`` if available in trajectories.

Usage
-----
  conda run -n cs234_rl python scripts/plot_inventory_paths.py \\
      --rl-run-dir runs/dqn_abides_final \\
      --baseline-run-dir runs/twap_abides_final \\
      --Q0 1000 --T 60 \\
      --out reports/figures/inventory_path.png

  # High-frequency view (uses timestamp_ns from trajectories.csv)
  conda run -n cs234_rl python scripts/plot_inventory_paths.py \\
      --rl-run-dir runs/dqn_hf_eval \\
      --baseline-run-dir runs/twap_hf_eval \\
      --x-axis time --time-unit ms \\
      --Q0 1000 \\
      --out reports/figures/inventory_path_hf_ms.png
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

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


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


def _infer_rl_label(rl_run_dir: str) -> str:
    episodes_df = load_episodes(rl_run_dir)
    if "strategy" in episodes_df.columns and not episodes_df["strategy"].dropna().empty:
        return _pretty_strategy_name(str(episodes_df["strategy"].dropna().iloc[0]))
    return "RL"


def _add_elapsed_time(traj: pd.DataFrame) -> pd.DataFrame:
    """Add per-row elapsed time in ns from each episode start."""
    if "timestamp_ns" not in traj.columns:
        raise ValueError("Trajectories missing 'timestamp_ns'. Re-run with updated logging.")
    out = traj.copy()
    out["timestamp_ns"] = pd.to_numeric(out["timestamp_ns"], errors="coerce")
    out = out.dropna(subset=["timestamp_ns"])
    out = out[out["timestamp_ns"] >= 0]
    if out.empty:
        raise ValueError("No valid timestamp_ns values found in trajectories.")
    out["elapsed_ns"] = out["timestamp_ns"] - out.groupby("episode")["timestamp_ns"].transform("min")
    return out


def _build_mean_path(
    traj: pd.DataFrame,
    Q0: float,
    T: int,
    eval_frac: float | None = None,
    x_axis: str = "step",
    time_unit: str = "ms",
) -> pd.DataFrame:
    """Mean remaining-inventory path anchored at (x=0, Q0)."""
    if eval_frac is not None:
        eps = sorted(traj["episode"].unique())
        cutoff = eps[int((1 - eval_frac) * len(eps))]
        traj = traj[traj["episode"] >= cutoff]

    if x_axis == "step":
        full_t = pd.Index(range(1, int(T) + 1), name="t")
        per_episode = []
        for _, ep_df in traj.groupby("episode"):
            ep_df = ep_df.sort_values("t")
            series = ep_df.set_index("t")["remaining_inventory"].astype(float)
            if series.empty:
                continue
            terminal_inv = float(series.iloc[-1])
            series = series.reindex(full_t)
            series = series.ffill().fillna(float(Q0))
            series = series.fillna(terminal_inv)
            if terminal_inv <= 1e-8:
                series = series.fillna(0.0)
            per_episode.append(series)

        if not per_episode:
            raise ValueError("No trajectory rows available to build the inventory path.")

        mean_path = pd.concat(per_episode, axis=1).mean(axis=1).reset_index()
        mean_path.columns = ["t", "remaining_inventory"]
        # Anchor at t=0
        path = pd.concat(
            [pd.DataFrame({"x": [0.0], "remaining_inventory": [float(Q0)]}),
             mean_path.rename(columns={"t": "x"})[["x", "remaining_inventory"]]],
            ignore_index=True,
        )
        # Extend to T with 0 if it terminates early
        if path["x"].max() < T:
            tail = pd.DataFrame({"x": [float(T)], "remaining_inventory": [0.0]})
            path = pd.concat([path, tail], ignore_index=True)
        return path

    traj_time = _add_elapsed_time(traj)
    scale = 1e9 if time_unit == "s" else 1e6
    mean_path = (
        traj_time.groupby("t", as_index=False)
        .agg(remaining_inventory=("remaining_inventory", "mean"),
             elapsed_ns=("elapsed_ns", "mean"))
        .sort_values("t")
    )
    mean_path["x"] = mean_path["elapsed_ns"] / scale
    path = pd.concat(
        [pd.DataFrame({"x": [0.0], "remaining_inventory": [float(Q0)]}),
         mean_path[["x", "remaining_inventory"]]],
        ignore_index=True,
    )

    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-run-dir",   required=True)
    p.add_argument("--baseline-run-dir", default=None)
    p.add_argument("--twap-run-dir", default=None, help="Backward-compatible alias for --baseline-run-dir")
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--rl-label", default=None)
    p.add_argument("--out",   default="reports/figures/inventory_path.png")
    p.add_argument("--Q0",  type=float, default=1000.0, help="Initial inventory")
    p.add_argument("--T",   type=int,   default=60,     help="Total time steps")
    p.add_argument("--x-axis", choices=["step", "time"], default="step",
                   help="Plot against decision step or elapsed time from timestamp_ns.")
    p.add_argument("--time-unit", choices=["ms", "s"], default="ms",
                   help="Time unit when --x-axis time.")
    p.add_argument("--eval-frac", type=float, default=0.2,
                   help="Fraction of last RL episodes used for mean path")
    args = p.parse_args()
    baseline_run_dir = args.baseline_run_dir or args.twap_run_dir
    if baseline_run_dir is None:
        raise ValueError("Provide --baseline-run-dir (or legacy --twap-run-dir).")
    rl_label = args.rl_label or _infer_rl_label(args.rl_run_dir)

    rl_traj   = load_trajectories(args.rl_run_dir)
    baseline_traj = load_trajectories(baseline_run_dir)

    rl_mean = _build_mean_path(
        rl_traj, args.Q0, args.T, eval_frac=args.eval_frac, x_axis=args.x_axis, time_unit=args.time_unit
    )
    baseline_mean = _build_mean_path(
        baseline_traj, args.Q0, args.T, x_axis=args.x_axis, time_unit=args.time_unit
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rl_mean["x"],   rl_mean["remaining_inventory"],
            color="#3A86FF", linewidth=2.5, label=rl_label)
    ax.plot(baseline_mean["x"], baseline_mean["remaining_inventory"],
            color="#FF6B6B", linewidth=2.5, label=args.baseline_label)

    if args.x_axis == "step":
        ax.set_xlim(0, args.T)
        x_label = "Step (t)"
    else:
        xmax = max(float(rl_mean["x"].max()), float(baseline_mean["x"].max()))
        ax.set_xlim(0, xmax * 1.02 if xmax > 0 else 1.0)
        x_label = f"Elapsed Time ({args.time_unit})"
    ax.set_ylim(0, args.Q0 * 1.05)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Remaining Inventory", fontsize=12)
    ax.set_title(f"Inventory Depletion: {rl_label} vs {args.baseline_label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
