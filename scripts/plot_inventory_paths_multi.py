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

from src.metrics.metrics import load_episodes, load_trajectories


COLORS = [
    "#3A86FF",
    "#FF6B6B",
    "#2A9D8F",
    "#F4A261",
    "#6A4C93",
    "#264653",
]


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


def _infer_label(run_dir: str) -> str:
    episodes_df = load_episodes(run_dir)
    if "strategy" in episodes_df.columns and not episodes_df["strategy"].dropna().empty:
        return _pretty_strategy_name(str(episodes_df["strategy"].dropna().iloc[0]))
    return Path(run_dir).name


def _build_mean_path(traj: pd.DataFrame, q0: float, horizon: int) -> pd.DataFrame:
    full_t = pd.Index(range(1, int(horizon) + 1), name="t")
    per_episode = []
    for _, ep_df in traj.groupby("episode"):
        ep_df = ep_df.sort_values("t")
        series = ep_df.set_index("t")["remaining_inventory"].astype(float)
        if series.empty:
            continue
        terminal_inv = float(series.iloc[-1])
        series = series.reindex(full_t)
        series = series.ffill().fillna(float(q0))
        if terminal_inv <= 1e-8:
            series = series.fillna(0.0)
        else:
            series = series.fillna(terminal_inv)
        per_episode.append(series)

    if not per_episode:
        raise ValueError("No trajectory rows available to build the inventory path.")

    mean_path = pd.concat(per_episode, axis=1).mean(axis=1).reset_index()
    mean_path.columns = ["t", "remaining_inventory"]
    return pd.concat(
        [
            pd.DataFrame({"t": [0.0], "remaining_inventory": [float(q0)]}),
            mean_path,
        ],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot inventory depletion curves for multiple strategies.")
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory. Repeat for each strategy.")
    parser.add_argument("--label", action="append", default=None, help="Optional display label. Repeat in the same order.")
    parser.add_argument("--out", type=str, default="reports/figures/inventory_all_strategies.png")
    parser.add_argument("--Q0", type=float, default=1000.0)
    parser.add_argument("--T", type=int, default=60)
    parser.add_argument("--tail-k", type=int, default=None, help="Optional: use only the last K episodes from each run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = list(args.run_dir)
    labels = list(args.label or [])
    if labels and len(labels) != len(run_dirs):
        raise ValueError("If provided, --label must be repeated exactly once per --run-dir.")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, run_dir in enumerate(run_dirs):
        traj = load_trajectories(run_dir)
        if args.tail_k is not None:
            episodes = load_episodes(run_dir)
            keep_eps = episodes["episode"].astype(int).sort_values().unique()[-max(1, int(args.tail_k)) :]
            traj = traj[traj["episode"].astype(int).isin(keep_eps)].copy()
        mean_path = _build_mean_path(traj, q0=args.Q0, horizon=args.T)
        label = labels[idx] if labels else _infer_label(run_dir)
        ax.plot(
            mean_path["t"],
            mean_path["remaining_inventory"],
            linewidth=2.4,
            color=COLORS[idx % len(COLORS)],
            label=label,
        )

    ax.set_xlim(0, args.T)
    ax.set_ylim(0, args.Q0 * 1.05)
    ax.set_xlabel("Step (t)")
    ax.set_ylabel("Remaining Inventory")
    ax.set_title("Inventory Depletion: All Strategies", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
