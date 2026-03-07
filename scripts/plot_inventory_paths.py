from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.metrics import load_trajectories, mean_inventory_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RL vs TWAP mean inventory trajectories.")
    parser.add_argument("--rl-run-dir", type=str, required=True, help="RL run directory.")
    parser.add_argument("--twap-run-dir", type=str, required=True, help="TWAP run directory.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/figures/inventory_path.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rl_traj = load_trajectories(args.rl_run_dir)
    twap_traj = load_trajectories(args.twap_run_dir)

    rl_path = mean_inventory_path(rl_traj)
    twap_path = mean_inventory_path(twap_traj)

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(rl_path["t"], rl_path["remaining_inventory"], label="RL", linewidth=2.0)
    plt.plot(twap_path["t"], twap_path["remaining_inventory"], label="TWAP", linewidth=2.0)
    plt.xlabel("Time step t")
    plt.ylabel("Remaining inventory")
    plt.title("Inventory Path: RL vs TWAP")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
