from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_episodes(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing episodes file: {path}")
    return pd.read_csv(path)


def load_trajectories(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / "trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectories file: {path}")
    return pd.read_csv(path)


def implementation_shortfall_by_episode(traj_df: pd.DataFrame) -> pd.Series:
    if "step_is" in traj_df.columns:
        return traj_df.groupby("episode")["step_is"].sum().astype(float)

    required = {"executed_qty", "mid_price", "exec_price", "episode"}
    if not required.issubset(set(traj_df.columns)):
        missing = sorted(required - set(traj_df.columns))
        raise ValueError(f"Trajectories missing columns for IS: {missing}")

    step_is = traj_df["executed_qty"] * (traj_df["mid_price"] - traj_df["exec_price"])
    return step_is.groupby(traj_df["episode"]).sum().astype(float)


def reward_by_episode(traj_df: pd.DataFrame) -> pd.Series:
    if "reward" not in traj_df.columns or "episode" not in traj_df.columns:
        raise ValueError("Trajectories must include 'episode' and 'reward'.")
    return traj_df.groupby("episode")["reward"].sum().astype(float)


def completion_rate(episodes_df: pd.DataFrame) -> float:
    if "completed" not in episodes_df.columns:
        raise ValueError("Episodes must include 'completed'.")
    return float(episodes_df["completed"].astype(float).mean())


def mean_inventory_path(traj_df: pd.DataFrame) -> pd.DataFrame:
    required = {"t", "remaining_inventory"}
    if not required.issubset(set(traj_df.columns)):
        missing = sorted(required - set(traj_df.columns))
        raise ValueError(f"Trajectories missing columns for inventory path: {missing}")

    out = (
        traj_df.groupby("t", as_index=False)["remaining_inventory"]
        .mean()
        .sort_values("t")
        .reset_index(drop=True)
    )
    return out


def summarize_run(run_dir: str | Path) -> dict[str, float]:
    episodes = load_episodes(run_dir)
    traj = load_trajectories(run_dir)

    is_per_ep = implementation_shortfall_by_episode(traj)
    reward_per_ep = reward_by_episode(traj)

    summary = {
        "episodes": float(len(episodes)),
        "mean_reward": float(reward_per_ep.mean()),
        "mean_is": float(is_per_ep.mean()),
        "p95_is": float(is_per_ep.quantile(0.95)),
        "completion_rate": completion_rate(episodes),
    }
    if "forced_qty" in traj.columns and "executed_qty" in traj.columns:
        total_forced = float(traj["forced_qty"].sum())
        total_exec = float(traj["executed_qty"].sum())
        summary["forced_liq_share"] = (total_forced / total_exec) if total_exec > 0 else 0.0
    return summary
