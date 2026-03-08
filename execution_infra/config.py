"""
Centralized configuration for the Optimal Execution environment.

All tunable hyperparameters live here so agents and experiments
can import a single EnvConfig object (or override fields via kwargs).

Set ``use_abides_replay=True`` (and point ``npz_path`` at a features.npz)
to swap the synthetic MarketSimulator for real ABIDES replay data.
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment hyper-parameters for the optimal-execution task."""

    # ── Trading task ──────────────────────────────────────────────
    total_inventory: int = 1000        # Total shares to liquidate
    n_steps: int = 60                  # Number of decision steps per episode

    # ── Action space ──────────────────────────────────────────────
    # Discrete actions: fraction of *remaining* inventory to sell
    # e.g. [0, 0.25, 0.5, 0.75, 1.0]  →  5 actions
    n_actions: int = 5

    # ── Market micro-structure ────────────────────────────────────
    initial_price: float = 100.0       # Starting mid-price ($)
    tick_size: float = 0.01            # Minimum price increment
    volatility: float = 0.002         # Per-step mid-price volatility (σ)
    spread_mean: float = 0.05          # Mean bid-ask spread ($)
    spread_std: float = 0.01           # Std-dev of spread noise
    lob_depth_mean: int = 500          # Mean volume at each LOB level
    lob_depth_std: int = 100           # Std-dev of LOB level volume noise
    n_lob_levels: int = 3              # Number of top LOB levels to track

    # ── Market impact ─────────────────────────────────────────────
    eta: float = 2.5e-6                # Temporary impact coefficient
    gamma_perm: float = 2.5e-7         # Permanent impact coefficient

    # ── Reward shaping ────────────────────────────────────────────
    lambda_penalty: float = 1.0        # Terminal penalty weight for leftover inventory

    # ── Random seed ───────────────────────────────────────────────
    seed: int | None = None            # Reproducibility (None → random)

    # ── ABIDES replay ────────────────────────────────────────────
    use_abides_replay: bool = False    # If True, use AbidesReplayEnv instead
    npz_path: str = "data/features.npz"  # Path to features.npz

    # ── Derived helpers ───────────────────────────────────────────
    @property
    def action_fractions(self) -> list[float]:
        """Return the fraction-of-remaining-inventory for each action."""
        if self.n_actions == 1:
            return [1.0]
        return [i / (self.n_actions - 1) for i in range(self.n_actions)]
