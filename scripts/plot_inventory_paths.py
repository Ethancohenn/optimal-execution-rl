"""
Plot mean inventory depletion paths for RL and TWAP on the same axes.
Both curves are anchored to (t=0, Q0) and extended to the full horizon T.

Usage
-----
  conda run -n cs234_rl python scripts/plot_inventory_paths.py \\
      --rl-run-dir   runs/dqn_abides_final \\
      --twap-run-dir runs/twap_abides_final \\
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-run-dir",   required=True)
    p.add_argument("--twap-run-dir", required=True)
    p.add_argument("--out",   default="reports/figures/inventory_path.png")
    p.add_argument("--Q0",  type=float, default=1000.0, help="Initial inventory")
    p.add_argument("--T",   type=int,   default=60,     help="Total time steps")
    p.add_argument("--eval-frac", type=float, default=0.2,
                   help="Fraction of last DQN episodes used for mean path")
    args = p.parse_args()

    rl_traj   = load_trajectories(args.rl_run_dir)
    twap_traj = load_trajectories(args.twap_run_dir)

    rl_mean   = _build_mean_path(rl_traj,   args.Q0, args.T, eval_frac=args.eval_frac)
    twap_mean = _build_mean_path(twap_traj, args.Q0, args.T)  # all episodes (deterministic)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rl_mean["t"],   rl_mean["remaining_inventory"],
            color="#3A86FF", linewidth=2.5, label=f"DQN")
    ax.plot(twap_mean["t"], twap_mean["remaining_inventory"],
            color="#FF6B6B", linewidth=2.5, label="TWAP")

    ax.set_xlim(0, args.T)
    ax.set_ylim(0, args.Q0 * 1.05)
    ax.set_xlabel("Step (t)", fontsize=12)
    ax.set_ylabel("Remaining Inventory", fontsize=12)
    ax.set_title("Inventory Depletion: DQN vs TWAP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
