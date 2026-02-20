"""
Lightweight Limit-Order-Book (LOB) market simulator.

Models mid-price evolution (GBM + permanent impact), bid-ask spread,
order-book depth, and temporary & permanent market-impact of trades.
This is intentionally decoupled from the Gym wrapper so it can be
tested and extended independently.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from execution_infra.config import EnvConfig


@dataclass
class LOBSnapshot:
    """Compact representation of the current order-book state."""
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    # Volume at best 3 levels on each side  (bid_vol + ask_vol)
    bid_volumes: np.ndarray      # shape (n_levels,)
    ask_volumes: np.ndarray      # shape (n_levels,)

    @property
    def total_volume(self) -> float:
        """Sum of volume across all tracked levels (both sides)."""
        return float(self.bid_volumes.sum() + self.ask_volumes.sum())


class MarketSimulator:
    """
    Single-asset LOB simulator with market-impact mechanics.

    Price dynamics
    ──────────────
    Mid price follows geometric Brownian motion *plus* a permanent
    downward shift each time the agent sells:

        P_mid(t+1) = P_mid(t) * exp(σ·Z)  −  γ · cumulative_volume

    Execution price
    ───────────────
    When the agent sells *q* shares the execution (fill) price is:

        P_exec = P_mid · (1 − η·q − γ·total_volume_traded)

    where η is the *temporary* impact coefficient and γ is the
    *permanent* impact coefficient.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.rng: np.random.Generator = np.random.default_rng(self.cfg.seed)

        # Will be set on reset()
        self.mid_price: float = 0.0
        self.cumulative_volume: float = 0.0
        self._lob: LOBSnapshot | None = None

    # ── public API ────────────────────────────────────────────────

    def reset(self) -> LOBSnapshot:
        """Initialise (or re-initialise) the market and return the first LOB snapshot."""
        self.mid_price = self.cfg.initial_price
        self.cumulative_volume = 0.0
        self._lob = self._generate_lob()
        return self._lob

    def step(self, quantity: int) -> tuple[float, LOBSnapshot]:
        """
        Execute a sell of *quantity* shares and advance the market by one tick.

        Returns
        -------
        execution_price : float
            The average fill price received by the agent.
        lob : LOBSnapshot
            The new LOB snapshot *after* the trade and price evolution.
        """
        if quantity < 0:
            raise ValueError("quantity must be non-negative")

        # ── 1. compute execution price ────────────────────────────
        if quantity == 0:
            execution_price = self.mid_price          # no trade → reference
        else:
            # Market impact formula:
            #   P_exec = P_mid * (1 - η·q - γ·cumulative_volume)
            impact = (
                self.cfg.eta * quantity
                + self.cfg.gamma_perm * self.cumulative_volume
            )
            execution_price = self.mid_price * (1.0 - impact)
            # Ensure price doesn't go negative
            execution_price = max(execution_price, self.cfg.tick_size)

        # ── 2. update permanent state ─────────────────────────────
        self.cumulative_volume += quantity

        # ── 3. evolve mid-price (GBM + permanent impact shift) ────
        #   dP = P · σ · Z  (log-normal step)
        z = self.rng.standard_normal()
        drift = -self.cfg.gamma_perm * quantity  # permanent price impact of this trade
        log_return = self.cfg.volatility * z + drift
        self.mid_price *= np.exp(log_return)
        # Snap to tick grid
        self.mid_price = round(self.mid_price / self.cfg.tick_size) * self.cfg.tick_size
        self.mid_price = max(self.mid_price, self.cfg.tick_size)

        # ── 4. regenerate LOB around new mid-price ────────────────
        self._lob = self._generate_lob()
        return execution_price, self._lob

    @property
    def lob(self) -> LOBSnapshot:
        if self._lob is None:
            raise RuntimeError("Call reset() before accessing LOB.")
        return self._lob

    # ── private helpers ───────────────────────────────────────────

    def _generate_lob(self) -> LOBSnapshot:
        """Sample a fresh LOB snapshot centred on the current mid-price."""
        spread = max(
            self.cfg.tick_size,
            self.rng.normal(self.cfg.spread_mean, self.cfg.spread_std),
        )
        half_spread = spread / 2.0
        best_bid = round((self.mid_price - half_spread) / self.cfg.tick_size) * self.cfg.tick_size
        best_ask = round((self.mid_price + half_spread) / self.cfg.tick_size) * self.cfg.tick_size

        # Volume at each level ~ Normal(mean, std), clipped to positive
        bid_volumes = np.clip(
            self.rng.normal(self.cfg.lob_depth_mean, self.cfg.lob_depth_std, size=self.cfg.n_lob_levels),
            1, None,
        ).astype(np.float64)
        ask_volumes = np.clip(
            self.rng.normal(self.cfg.lob_depth_mean, self.cfg.lob_depth_std, size=self.cfg.n_lob_levels),
            1, None,
        ).astype(np.float64)

        return LOBSnapshot(
            mid_price=self.mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=best_ask - best_bid,
            bid_volumes=bid_volumes,
            ask_volumes=ask_volumes,
        )
