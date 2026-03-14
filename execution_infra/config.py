"""Centralized configuration for the optimal-execution environments."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment hyper-parameters for the optimal-execution task."""

    # Trading task
    total_inventory: int = 1_000
    n_steps: int = 60
    n_actions: int = 5
    force_liquidation: bool = True

    # Market microstructure
    initial_price: float = 100.0
    tick_size: float = 0.01
    spread_mean: float = 0.05
    spread_std: float = 0.01
    min_spread_ticks: int = 1
    max_spread_ticks: int = 8
    lob_depth_mean: int = 500
    lob_depth_std: int = 100
    n_lob_levels: int = 5
    depth_decay: float = 0.35
    obs_depth_levels: int = 3

    # Exogenous order flow
    market_lot_size: int = 25
    market_order_intensity: float = 2.0
    limit_order_intensity: float = 1.2
    cancellation_rate: float = 0.04
    quote_improve_prob: float = 0.15
    volatility: float = 0.02
    fundamental_mean_reversion: float = 0.10
    order_flow_memory: float = 0.25

    # Agent impact
    eta: float = 2.5e-6
    gamma_perm: float = 2.5e-7
    agent_permanent_impact: float = 1.5

    # Reward shaping / termination
    lambda_penalty: float = 1.0
    urgency_penalty: float = 2.0

    # Misc
    seed: int | None = None
    use_abides_replay: bool = False
    npz_path: str = "data/features.npz"

    @property
    def action_fractions(self) -> list[float]:
        """Return the fraction-of-remaining-inventory for each action."""
        if self.n_actions == 1:
            return [1.0]
        return [i / (self.n_actions - 1) for i in range(self.n_actions)]
