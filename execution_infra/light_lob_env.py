"""
Lightweight Gymnasium LOB environment for fast RL iteration.

This environment is intentionally simpler than ABIDES but still keeps
core execution mechanics:
- stochastic mid-price dynamics
- spread and top-of-book liquidity
- market impact from trading
- terminal penalty for unfinished liquidation

Observation (6-dim float32):
  [inv_norm, time_norm, mid_norm, spread_norm, bid_vol_norm, obi]
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LightLOBEnv(gym.Env):
    """A learnable and fast execution environment with simple LOB dynamics."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_steps: int = 60,
        total_inventory: int = 1_000,
        n_actions: int = 5,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        eta: float = 2.5e-6,
        gamma_perm: float = 2.5e-7,
        lambda_penalty: float = 1.0,
        urgency_coef: float = 10.0,
        spread_mean: float = 0.02,
        spread_std: float = 0.004,
        base_depth: float = 1_500.0,
        signal_rho: float = 0.85,
        signal_noise: float = 0.12,
        drift_strength: float = 0.0015,
        return_noise: float = 0.0009,
        participation_impact: float = 0.04,
        depth_noise: float = 0.15,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.total_inventory = int(total_inventory)
        self.n_actions = int(n_actions)
        self.initial_price = float(initial_price)
        self.tick_size = float(tick_size)
        self.eta = float(eta)
        self.gamma_perm = float(gamma_perm)
        self.lambda_penalty = float(lambda_penalty)
        self.urgency_coef = float(urgency_coef)
        self.spread_mean = float(spread_mean)
        self.spread_std = float(spread_std)
        self.base_depth = float(base_depth)
        self.signal_rho = float(signal_rho)
        self.signal_noise = float(signal_noise)
        self.drift_strength = float(drift_strength)
        self.return_noise = float(return_noise)
        self.participation_impact = float(participation_impact)
        self.depth_noise = float(depth_noise)
        self.render_mode = render_mode

        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.total_inventory <= 0:
            raise ValueError("total_inventory must be positive.")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive.")

        if self.n_actions == 1:
            self._fractions = [1.0]
        else:
            self._fractions = [i / (self.n_actions - 1) for i in range(self.n_actions)]

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)

        # Runtime state (set in reset).
        self._step = 0
        self._inventory = 0
        self._benchmark_price = 0.0
        self._cumulative_volume = 0.0
        self._regime = 0.0
        self._signal = 0.0
        self._mid_price = 0.0
        self._spread = 0.0
        self._bid_vol_1 = 0.0
        self._ask_vol_1 = 0.0

        # Compatibility fields used by existing scripts.
        self._mean_spread = self.spread_mean
        self._mean_bid_vol = self.base_depth
        self.trades: list[dict] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step = 0
        self._inventory = self.total_inventory
        self._cumulative_volume = 0.0
        self.trades = []

        # Episode regime: +1 (upward tendency) or -1 (downward tendency).
        self._regime = float(self._rng.choice([-1.0, 1.0]))
        self._signal = float(np.clip(0.65 * self._regime + self._rng.normal(0.0, 0.15), -1.0, 1.0))
        self._mid_price = float(max(self.tick_size, self.initial_price * (1.0 + self._rng.normal(0.0, 0.002))))
        self._benchmark_price = self._mid_price
        self._update_lob_from_signal()

        return self._obs(), self._info(exec_price=float("nan"), mid_price=self._mid_price, executed_qty=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        frac = self._fractions[int(action)]
        quantity = int(np.round(frac * self._inventory))
        if action > 0 and self._inventory > 0 and quantity == 0:
            quantity = 1
        if action > 0 and self._inventory > 0 and self._step == (self.n_steps - 1):
            quantity = self._inventory

        return self._transition(quantity)

    def step_with_qty(self, quantity: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute an exact quantity (used by TWAP/immediate/last-minute baselines)."""
        return self._transition(int(quantity))

    def render(self) -> None:
        sold = self.total_inventory - self._inventory
        pct = 100.0 * sold / max(1, self.total_inventory)
        print(
            f"Step {self._step:>3}/{self.n_steps} | "
            f"Inventory {self._inventory:>6} ({pct:5.1f}% sold) | "
            f"Mid {self._mid_price:.4f} | Spread {self._spread:.4f} | OBI {self._obi():.3f}"
        )

    def _transition(self, quantity: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        quantity = max(0, min(int(quantity), self._inventory))

        mid_before = float(self._mid_price)
        spread_before = float(self._spread)
        bid_liquidity = max(1.0, float(self._bid_vol_1))

        if quantity == 0:
            exec_price = mid_before
            step_is = 0.0
        else:
            participation = quantity / bid_liquidity
            impact_frac = (
                self.eta * quantity
                + self.gamma_perm * self._cumulative_volume
                + self.participation_impact * participation
            )
            impact_frac = float(np.clip(impact_frac, 0.0, 0.95))
            exec_price = mid_before * (1.0 - impact_frac) - 0.5 * spread_before
            exec_price = float(max(exec_price, self.tick_size))
            step_is = float(quantity * (mid_before - exec_price))

        # Reward is aligned with maximising sale value relative to arrival price
        # while discouraging waiting near the deadline.
        reward = float((exec_price - self._benchmark_price) * quantity)
        inv_fraction = self._inventory / max(1.0, float(self.total_inventory))
        time_pressure = self._step / max(1.0, float(self.n_steps - 1))
        reward -= float(self.urgency_coef * time_pressure * inv_fraction)

        self._inventory -= quantity
        self._cumulative_volume += quantity
        self._step += 1

        if quantity > 0:
            self.trades.append(
                {
                    "step": self._step,
                    "quantity": quantity,
                    "exec_price": exec_price,
                    "mid_price": mid_before,
                    "step_is": step_is,
                    "reward": reward,
                }
            )

        terminated = self._inventory <= 0
        truncated = (self._step >= self.n_steps) and (self._inventory > 0)
        if truncated:
            penalty = float(self.lambda_penalty * self._inventory * self._benchmark_price)
            reward -= penalty

        if not (terminated or truncated):
            self._evolve_market(executed_qty=quantity)

        obs = self._obs()
        info = self._info(exec_price=exec_price, mid_price=mid_before, executed_qty=quantity, step_is=step_is)

        if self.render_mode == "human" and (terminated or truncated):
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _evolve_market(self, executed_qty: int) -> None:
        signal_eps = self._rng.normal(0.0, self.signal_noise)
        self._signal = float(
            np.clip(
                self.signal_rho * self._signal + (1.0 - self.signal_rho) * self._regime + signal_eps,
                -1.0,
                1.0,
            )
        )

        ret = (
            self.drift_strength * self._signal
            + self.return_noise * self._rng.standard_normal()
            - 0.20 * self.gamma_perm * float(executed_qty)
        )
        self._mid_price *= float(np.exp(ret))
        self._mid_price = float(max(self.tick_size, round(self._mid_price / self.tick_size) * self.tick_size))

        self._update_lob_from_signal()

    def _update_lob_from_signal(self) -> None:
        spread_noise = abs(self._rng.normal(0.0, self.spread_std))
        spread_level = self.spread_mean * (1.0 + 0.45 * (1.0 - abs(self._signal)))
        self._spread = float(max(self.tick_size, spread_level + spread_noise))

        bid_mult = 1.0 + 0.60 * self._signal
        ask_mult = 1.0 - 0.60 * self._signal
        bid_noise = float(max(0.20, 1.0 + self.depth_noise * self._rng.standard_normal()))
        ask_noise = float(max(0.20, 1.0 + self.depth_noise * self._rng.standard_normal()))

        self._bid_vol_1 = float(max(1.0, self.base_depth * bid_mult * bid_noise))
        self._ask_vol_1 = float(max(1.0, self.base_depth * ask_mult * ask_noise))

    def _obi(self) -> float:
        denom = max(self._bid_vol_1 + self._ask_vol_1, 1.0)
        return float(np.clip(self._bid_vol_1 / denom, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        inv_norm = self._inventory / max(1.0, float(self.total_inventory))
        time_norm = 1.0 - self._step / max(1.0, float(self.n_steps))
        mid_norm = self._mid_price / max(self._benchmark_price, 1e-8)
        spread_norm = self._spread / max(self.spread_mean, 1e-8)
        bid_vol_norm = self._bid_vol_1 / max(self.base_depth, 1e-8)
        obi = self._obi()
        return np.array([inv_norm, time_norm, mid_norm, spread_norm, bid_vol_norm, obi], dtype=np.float32)

    def _info(
        self,
        exec_price: float = float("nan"),
        mid_price: float = float("nan"),
        executed_qty: int = 0,
        step_is: float = 0.0,
    ) -> dict:
        remaining = int(self._inventory)
        sold = int(self.total_inventory - remaining)
        return {
            "inventory_remaining": remaining,
            "remaining_inventory": float(remaining),
            "shares_sold": sold,
            "step": int(self._step),
            "t": int(self._step),
            "exec_price": float(exec_price),
            "mid_price": float(mid_price),
            "executed_qty": int(executed_qty),
            "forced_qty": 0.0,
            "step_is": float(step_is),
            "benchmark_price": float(self._benchmark_price),
            "spread": float(self._spread),
            "bid_vol_1": float(self._bid_vol_1),
            "order_book_imbalance": float(self._obi()),
            "completion_rate": sold / max(1.0, float(self.total_inventory)),
            "regime": float(self._regime),
        }
