from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL + TWAP and generate milestone plots.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--rl-run-name", type=str, default=None)
    parser.add_argument("--twap-run-name", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default="reports/figures")
    parser.add_argument("--rl-alpha", type=float, default=0.1)
    parser.add_argument("--rl-gamma", type=float, default=1.0)
    parser.add_argument("--rl-epsilon-start", type=float, default=1.0)
    parser.add_argument("--rl-epsilon-end", type=float, default=0.01)
    parser.add_argument(
        "--rl-epsilon-decay-episodes",
        type=int,
        default=None,
        help="Defaults to --episodes if omitted.",
    )
    parser.add_argument("--rl-inv-bins", type=int, default=10)
    parser.add_argument("--rl-time-bins", type=int, default=10)
    parser.add_argument(
        "--summary-tail-k",
        type=int,
        default=None,
        help="If set, IS summary/boxplot uses only the last K episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directories if names already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rl_epsilon_decay_episodes is None:
        args.rl_epsilon_decay_episodes = int(args.episodes)
    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.rl_run_name is None:
        args.rl_run_name = f"run_rl_stub_{stamp}"
    if args.twap_run_name is None:
        args.twap_run_name = f"run_twap_stub_{stamp}"
    rl_run_dir = str(root / args.base_run_dir / args.rl_run_name)
    twap_run_dir = str(root / args.base_run_dir / args.twap_run_name)
    figure_dir = Path(root / args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rl_cmd = [
        py,
        "-m",
        "src.run_stub",
        "--episodes",
        str(args.episodes),
        "--T",
        str(args.T),
        "--Q0",
        str(args.Q0),
        "--seed",
        str(args.seed),
        "--alpha",
        str(args.rl_alpha),
        "--gamma",
        str(args.rl_gamma),
        "--epsilon-start",
        str(args.rl_epsilon_start),
        "--epsilon-end",
        str(args.rl_epsilon_end),
        "--epsilon-decay-episodes",
        str(args.rl_epsilon_decay_episodes),
        "--inv-bins",
        str(args.rl_inv_bins),
        "--time-bins",
        str(args.rl_time_bins),
        "--base-run-dir",
        args.base_run_dir,
        "--run-name",
        args.rl_run_name,
    ]
    if args.overwrite:
        rl_cmd.append("--overwrite")
    run_cmd(rl_cmd)

    twap_cmd = [
        py,
        "-m",
        "src.baselines.twap",
        "--episodes",
        str(args.episodes),
        "--T",
        str(args.T),
        "--Q0",
        str(args.Q0),
        "--seed",
        str(args.seed),
        "--base-run-dir",
        args.base_run_dir,
        "--run-name",
        args.twap_run_name,
    ]
    if args.overwrite:
        twap_cmd.append("--overwrite")
    run_cmd(twap_cmd)

    run_cmd(
        [
            py,
            "scripts/plot_training_curve.py",
            "--run-dir",
            rl_run_dir,
            "--out",
            str(figure_dir / "training_curve.png"),
        ]
    )

    run_cmd(
        [
            py,
            "scripts/plot_inventory_paths.py",
            "--rl-run-dir",
            rl_run_dir,
            "--twap-run-dir",
            twap_run_dir,
            "--out",
            str(figure_dir / "inventory_path.png"),
        ]
    )

    run_cmd(
        (
            [
            py,
            "scripts/plot_implementation_shortfall.py",
            "--rl-run-dir",
            rl_run_dir,
            "--twap-run-dir",
            twap_run_dir,
            "--out",
            str(figure_dir / "implementation_shortfall.png"),
            ]
            + (["--tail-k", str(args.summary_tail_k)] if args.summary_tail_k is not None else [])
        )
    )

    print("Milestone evaluation artifacts generated.")
    print(f"RL run: {rl_run_dir}")
    print(f"TWAP run: {twap_run_dir}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
