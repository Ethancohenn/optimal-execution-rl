"""
Plot mean inventory depletion paths for one or more runs on the same axes.
Each curve is anchored to (t=0, Q0) and extended to the full horizon T.

Usage (multi-run)
-----------------
  conda run -n cs234_rl python scripts/plot_inventory_paths.py \\
      --run-dirs runs/dqn_eval runs/ddqn_eval runs/twap_eval \\
      --labels DQN DDQN TWAP \\
      --Q0 1000 --T 60 --eval-frac 1.0 \\
      --out reports/figures/inventory_path_all.png

Usage (legacy 2-run mode)
-------------------------
  conda run -n cs234_rl python scripts/plot_inventory_paths.py \\
      --rl-run-dir runs/dqn_abides_final \\
      --baseline-run-dir runs/twap_abides_final \\
      --Q0 1000 --T 60 \\
      --out reports/figures/inventory_path.png
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

from src.metrics.metrics import load_trajectories

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _build_mean_path(traj: pd.DataFrame, Q0: float, T: int,
                     eval_frac: float | None = None) -> pd.DataFrame:
    """Mean remaining-inventory path anchored at (0, Q0) and padded to T."""
    if eval_frac is not None:
        eps = sorted(traj["episode"].unique())
        cutoff = eps[int((1 - eval_frac) * len(eps))]
        traj = traj[traj["episode"] >= cutoff]

    mean_path = (traj.groupby("t", as_index=False)["remaining_inventory"]
                 .mean().sort_values("t"))

    # Anchor at t=0
    t0 = pd.DataFrame({"t": [0], "remaining_inventory": [float(Q0)]})
    path = pd.concat([t0, mean_path], ignore_index=True)

    # Extend to T with 0 if it terminates early
    if path["t"].max() < T:
        tail = pd.DataFrame({"t": [T], "remaining_inventory": [0.0]})
        path = pd.concat([path, tail], ignore_index=True)

    return path


def _build_run_specs(args: argparse.Namespace) -> list[tuple[str, str, float | None]]:
    # New multi-run mode.
    if args.run_dirs is not None:
        if len(args.run_dirs) < 2:
            raise ValueError("Provide at least two paths in --run-dirs for comparison.")
        if args.labels is not None and len(args.labels) != len(args.run_dirs):
            raise ValueError("--labels count must match --run-dirs count.")
        labels = args.labels or [Path(d).name for d in args.run_dirs]
        return [(d, lbl, args.eval_frac) for d, lbl in zip(args.run_dirs, labels)]

    # Legacy 2-run mode.
    baseline_run_dir = args.baseline_run_dir or args.twap_run_dir
    if baseline_run_dir is None:
        raise ValueError(
            "Provide either --run-dirs (multi-run mode) or "
            "--baseline-run-dir/--twap-run-dir (legacy mode)."
        )
    if args.rl_run_dir is None:
        raise ValueError("In legacy mode, --rl-run-dir is required.")
    return [
        (args.rl_run_dir, args.rl_label, args.eval_frac),
        (baseline_run_dir, args.baseline_label, None),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    # Multi-run mode (recommended)
    p.add_argument("--run-dirs", nargs="+", default=None,
                   help="Run directories to overlay on one figure.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display labels for --run-dirs (same count/order).")

    # Legacy mode (backward compatibility)
    p.add_argument("--rl-run-dir", default=None)
    p.add_argument("--rl-label", default="DQN")
    p.add_argument("--baseline-run-dir", default=None)
    p.add_argument("--twap-run-dir", default=None, help="Backward-compatible alias for --baseline-run-dir")
    p.add_argument("--baseline-label", default="Baseline")

    p.add_argument("--out",   default="reports/figures/inventory_path.png")
    p.add_argument("--Q0",  type=float, default=1000.0, help="Initial inventory")
    p.add_argument("--T",   type=int,   default=60,     help="Total time steps")
    p.add_argument("--eval-frac", type=float, default=0.2,
                   help="Fraction of last episodes used per run in --run-dirs mode.")
    args = p.parse_args()

    run_specs = _build_run_specs(args)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = list(plt.get_cmap("tab10").colors)
    for idx, (run_dir, label, eval_frac) in enumerate(run_specs):
        traj = load_trajectories(run_dir)
        mean_path = _build_mean_path(traj, args.Q0, args.T, eval_frac=eval_frac)
        color = palette[idx % len(palette)]
        ax.plot(
            mean_path["t"],
            mean_path["remaining_inventory"],
            color=color,
            linewidth=2.2,
            label=label,
        )

    ax.set_xlim(0, args.T)
    ax.set_ylim(0, args.Q0 * 1.05)
    ax.set_xlabel("Step (t)", fontsize=12)
    ax.set_ylabel("Remaining Inventory", fontsize=12)
    if len(run_specs) == 2 and args.run_dirs is None:
        ax.set_title(
            f"Inventory Depletion: {run_specs[0][1]} vs {run_specs[1][1]}",
            fontsize=13,
            fontweight="bold",
        )
    else:
        ax.set_title("Inventory Depletion Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
