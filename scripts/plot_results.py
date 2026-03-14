"""
Plotting utilities for optimal-execution runs.

Three public functions
---------------------
  plot_training_curve(dqn_ep, out_path)
      Training reward + smoothed curve + epsilon overlay.

  plot_implementation_shortfall(dqn_ep, twap_ep, out_path)
      IS distribution comparison between DQN and TWAP.

  plot_inventory_paths(dqn_traj, twap_traj, out_path, n_samples)
      Overlaid inventory-depletion paths for sampled episodes.

CLI usage
---------
  python scripts/plot_results.py \\
      --dqn-dir  runs/dqn_abides_v2 \\
      --twap-dir runs/twap_abides_v1 \\
      --out-dir  reports/plots
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

DQN_COLOR  = "#3A86FF"
TWAP_COLOR = "#FF6B6B"


def _smooth(x: np.ndarray, w: int = 30) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    dqn_ep: pd.DataFrame,
    out_path: str = "reports/plots/training_curve.png",
) -> None:
    """Episode reward + smoothed curve + epsilon decay.

    Parameters
    ----------
    dqn_ep   : episodes.csv loaded as a DataFrame
    out_path : where to save the figure
    """
    r = dqn_ep["episode_reward"].values
    eps = dqn_ep["epsilon"].values

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Reward (primary axis)
    ax1.plot(r, color=DQN_COLOR, alpha=0.2, linewidth=0.8, label="Episode reward")
    ax1.plot(_smooth(r), color="#1B5299", linewidth=2.0, label="Smoothed (w=30)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward", color="#1B5299")
    ax1.tick_params(axis="y", labelcolor="#1B5299")

    # Epsilon (secondary axis)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(eps, color="#FF9F1C", linewidth=1.2, linestyle=":", alpha=0.8, label="ε (epsilon)")
    ax2.set_ylabel("ε", color="#FF9F1C")
    ax2.tick_params(axis="y", labelcolor="#FF9F1C")
    ax2.set_ylim(-0.05, 1.1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    ax1.set_title("DQN Training Curve")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Implementation shortfall
# ─────────────────────────────────────────────────────────────────────────────

def plot_implementation_shortfall(
    dqn_ep: pd.DataFrame,
    twap_ep: pd.DataFrame | None = None,
    out_path: str = "reports/plots/implementation_shortfall.png",
    eval_frac: float = 0.2,
) -> None:
    """Implementation shortfall (IS) distributions: DQN vs TWAP.

    Uses the last ``eval_frac`` fraction of DQN episodes as the
    evaluation window (after exploration has decayed).

    Parameters
    ----------
    dqn_ep    : DQN episodes.csv DataFrame
    twap_ep   : TWAP episodes.csv DataFrame (or None)
    out_path  : save path
    eval_frac : fraction of last episodes used for DQN evaluation
    """
    n = len(dqn_ep)
    dqn_eval = dqn_ep["episode_is"].iloc[int((1 - eval_frac) * n):]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(40, max(10, n // 20))

    ax.hist(dqn_eval.values, bins=bins, color=DQN_COLOR, alpha=0.75, density=True,
            label=f"DQN — last {int(eval_frac*100)}%  (n={len(dqn_eval)})")
    ax.axvline(dqn_eval.mean(), color="#1B5299", linestyle="--", linewidth=2,
               label=f"DQN mean = {dqn_eval.mean():.2f}")

    if twap_ep is not None and "episode_is" in twap_ep.columns:
        tw = twap_ep["episode_is"]
        ax.hist(tw.values, bins=bins, color=TWAP_COLOR, alpha=0.75, density=True,
                label=f"TWAP  (n={len(tw)})")
        ax.axvline(tw.mean(), color="#8B0000", linestyle="--", linewidth=2,
                   label=f"TWAP mean = {tw.mean():.2f}")

    ax.set_xlabel("Implementation Shortfall (IS)")
    ax.set_ylabel("Density")
    ax.set_title("Implementation Shortfall: DQN vs TWAP\n(lower IS = better execution)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inventory paths
# ─────────────────────────────────────────────────────────────────────────────

def plot_inventory_paths(
    dqn_traj: pd.DataFrame,
    twap_traj: pd.DataFrame | None = None,
    out_path: str = "reports/plots/inventory_paths.png",
    n_samples: int = 10,
    eval_frac: float = 0.2,
) -> None:
    """Overlay inventory-depletion paths for sampled episodes.

    Parameters
    ----------
    dqn_traj  : DQN trajectories.csv DataFrame
    twap_traj : TWAP trajectories.csv DataFrame (or None)
    out_path  : save path
    n_samples : number of DQN episodes to overlay
    eval_frac : sample from the last ``eval_frac`` fraction of DQN episodes
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # ── DQN sample paths ──────────────────────────────────────────
    all_eps = sorted(dqn_traj["episode"].unique())
    cutoff = all_eps[int((1 - eval_frac) * len(all_eps))]
    eval_eps = [e for e in all_eps if e >= cutoff]
    sampled = np.random.default_rng(0).choice(eval_eps, size=min(n_samples, len(eval_eps)), replace=False)

    for i, ep in enumerate(sampled):
        ep_df = dqn_traj[dqn_traj["episode"] == ep]
        # Prepend t=0, full inventory
        inv0 = ep_df["remaining_inventory"].iloc[0] + ep_df["executed_qty"].iloc[0]
        t_vals = np.concatenate([[0], ep_df["t"].values])
        inv_vals = np.concatenate([[inv0], ep_df["remaining_inventory"].values])
        ax.plot(t_vals, inv_vals, color=DQN_COLOR, alpha=0.4, linewidth=1.0,
                label="DQN episodes" if i == 0 else None)

    # ── TWAP mean path ────────────────────────────────────────────
    if twap_traj is not None:
        tw_mean = twap_traj.groupby("t")["remaining_inventory"].mean().reset_index()
        t0_inv = twap_traj.groupby("episode").apply(
            lambda g: g["remaining_inventory"].iloc[0] + g["executed_qty"].iloc[0]
        ).mean()
        t_vals = np.concatenate([[0], tw_mean["t"].values])
        inv_vals = np.concatenate([[t0_inv], tw_mean["remaining_inventory"].values])
        ax.plot(t_vals, inv_vals, color=TWAP_COLOR, linewidth=2.5,
                linestyle="--", label="TWAP (mean path)", zorder=5)

    # ── Linear benchmark ─────────────────────────────────────────
    max_t = dqn_traj["t"].max()
    max_inv = dqn_traj.groupby("episode").apply(
        lambda g: g["remaining_inventory"].iloc[0] + g["executed_qty"].iloc[0]
    ).mean()
    ax.plot([0, max_t], [max_inv, 0], color="grey", linewidth=1.5,
            linestyle=":", label="Linear (reference)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Remaining Inventory")
    ax.set_title(f"Inventory Depletion Paths\n(DQN: {len(sampled)} sampled eval episodes  |  TWAP: mean)")
    ax.legend(fontsize=9)
    ax.set_ylim(-50, max_inv * 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load(run_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found in {run_dir}")
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training curves, IS, and inventory paths.")
    p.add_argument("--dqn-dir",   required=True)
    p.add_argument("--twap-dir",  default=None)
    p.add_argument("--out-dir",   default="reports/plots")
    p.add_argument("--n-samples", type=int, default=10, help="Inventory path episodes to overlay")
    args = p.parse_args()

    dqn_ep   = _load(args.dqn_dir, "episodes.csv")
    dqn_traj = _load(args.dqn_dir, "trajectories.csv")
    twap_ep, twap_traj = None, None
    if args.twap_dir:
        twap_ep   = _load(args.twap_dir, "episodes.csv")
        twap_traj = _load(args.twap_dir, "trajectories.csv")

    print(f"\nGenerating plots → {args.out_dir}/")
    plot_training_curve(dqn_ep,
        out_path=os.path.join(args.out_dir, "training_curve.png"))
    plot_implementation_shortfall(dqn_ep, twap_ep,
        out_path=os.path.join(args.out_dir, "implementation_shortfall.png"))
    plot_inventory_paths(dqn_traj, twap_traj,
        out_path=os.path.join(args.out_dir, "inventory_paths.png"),
        n_samples=args.n_samples)


if __name__ == "__main__":
    main()
