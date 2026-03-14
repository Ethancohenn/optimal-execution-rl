"""
Tiny, fast Gymnasium LOB environment for sanity-checking learning.

Design goals:
- very lightweight dynamics (fast to run)
- clear, learnable structure (signal + liquidity regime)
- API-compatible with the existing training/baseline scripts
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TinyLOBEnv(gym.Env):
    """A minimal yet learnable execution environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_steps: int = 40,
        total_inventory: int = 1_000,
        n_actions: int = 5,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        spread_mean: float = 0.03,
        spread_std: float = 0.004,
        base_depth: float = 1_200.0,
        signal_rho: float = 0.90,
        signal_noise: float = 0.08,
        drift_strength: float = 0.0010,
        return_noise: float = 0.0005,
        depth_sensitivity: float = 0.55,
        eta: float = 0.35,
        gamma_perm: float = 0.12,
        lambda_penalty: float = 1.0,
        urgency_coef: float = 0.02,
        force_liquidation: bool = True,
        terminal_impact_mult: float = 2.5,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.total_inventory = int(total_inventory)
        self.n_actions = int(n_actions)
        self.initial_price = float(initial_price)
        self.tick_size = float(tick_size)
        self.spread_mean = float(spread_mean)
        self.spread_std = float(spread_std)
        self.base_depth = float(base_depth)
        self.signal_rho = float(signal_rho)
        self.signal_noise = float(signal_noise)
        self.drift_strength = float(drift_strength)
        self.return_noise = float(return_noise)
        self.depth_sensitivity = float(depth_sensitivity)
        self.eta = float(eta)
        self.gamma_perm = float(gamma_perm)
        self.lambda_penalty = float(lambda_penalty)
        self.urgency_coef = float(urgency_coef)
        self.force_liquidation = bool(force_liquidation)
        self.terminal_impact_mult = float(terminal_impact_mult)
        self.render_mode = render_mode

        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.total_inventory <= 0:
            raise ValueError("total_inventory must be positive.")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        if self.base_depth <= 0:
            raise ValueError("base_depth must be positive.")
        if self.signal_rho < 0.0 or self.signal_rho >= 1.0:
            raise ValueError("signal_rho must be in [0, 1).")

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

        # Hidden episode regime that drives a low-noise predictable signal.
        self._regime = float(self._rng.choice([-1.0, 1.0]))
        self._signal = float(np.clip(0.75 * self._regime + self._rng.normal(0.0, 0.12), -1.0, 1.0))
        self._mid_price = float(max(self.tick_size, self.initial_price * (1.0 + self._rng.normal(0.0, 0.0015))))
        self._benchmark_price = self._mid_price
        self._update_lob_from_signal()

        info = self._info(exec_price=float("nan"), mid_price=self._mid_price, executed_qty=0, forced_qty=0, step_is=0.0)
        return self._obs(), info

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
        """Execute an exact quantity (used by fixed-schedule baselines)."""
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
        exec_qty = int(quantity)
        exec_price = float("nan")
        forced_qty = 0
        step_is = 0.0

        if exec_qty > 0:
            exec_price = self._execution_price(mid_before, exec_qty, terminal=False)
            step_is = float(exec_qty * (mid_before - exec_price))

        self._inventory -= exec_qty
        self._cumulative_volume += exec_qty
        self._step += 1

        terminated = False
        truncated = False
        terminal_penalty = 0.0

        if self._step >= self.n_steps and self._inventory > 0:
            if self.force_liquidation:
                forced_qty = int(self._inventory)
                forced_price = self._execution_price(mid_before, forced_qty, terminal=True)
                forced_step_is = float(forced_qty * (mid_before - forced_price))
                step_is += forced_step_is
                exec_qty += forced_qty
                self._cumulative_volume += forced_qty
                self._inventory = 0
            else:
                terminal_penalty = float(self.lambda_penalty * self._inventory * self._benchmark_price)
                truncated = True

        terminated = self._inventory <= 0

        # A small holding penalty makes waiting too long suboptimal.
        inv_fraction = self._inventory / max(1.0, float(self.total_inventory))
        time_pressure = (self._step - 1) / max(1.0, float(self.n_steps - 1))
        hold_penalty = float(self.urgency_coef * time_pressure * inv_fraction)
        reward = float(-step_is - hold_penalty - terminal_penalty)

        if exec_qty > 0:
            # Equivalent average execution price across action + any forced liquidation.
            exec_price = float(mid_before - (step_is / max(1, exec_qty)))
            self.trades.append(
                {
                    "step": self._step,
                    "quantity": int(exec_qty),
                    "forced_qty": int(forced_qty),
                    "exec_price": exec_price,
                    "mid_price": mid_before,
                    "step_is": float(step_is),
                    "reward": float(reward),
                }
            )

        if not (terminated or truncated):
            self._evolve_market(executed_qty=exec_qty)

        obs = self._obs()
        info = self._info(
            exec_price=exec_price,
            mid_price=mid_before,
            executed_qty=exec_qty,
            forced_qty=forced_qty,
            step_is=step_is,
        )

        if self.render_mode == "human" and (terminated or truncated):
            self.render()

        return obs, reward, bool(terminated), bool(truncated), info

    def _execution_price(self, mid_price: float, quantity: int, terminal: bool) -> float:
        depth = max(1.0, float(self._bid_vol_1))
        participation = quantity / depth
        cumulative_frac = self._cumulative_volume / max(1.0, float(self.total_inventory))
        impact_dollars = self.eta * participation + self.gamma_perm * cumulative_frac
        if terminal:
            impact_dollars *= self.terminal_impact_mult

        price = mid_price - 0.5 * self._spread - impact_dollars
        return float(max(self.tick_size, price))

    def _evolve_market(self, executed_qty: int) -> None:
        signal_eps = self._rng.normal(0.0, self.signal_noise)
        self._signal = float(
            np.clip(
                self.signal_rho * self._signal + (1.0 - self.signal_rho) * self._regime + signal_eps,
                -1.0,
                1.0,
            )
        )

        # Small sell pressure from the agent's own execution.
        sell_pressure = 0.20 * float(executed_qty) / max(1.0, float(self.total_inventory))
        log_ret = self.drift_strength * self._signal + self.return_noise * self._rng.standard_normal() - sell_pressure
        self._mid_price *= float(np.exp(log_ret))
        self._mid_price = float(max(self.tick_size, round(self._mid_price / self.tick_size) * self.tick_size))

        self._update_lob_from_signal()

    def _update_lob_from_signal(self) -> None:
        spread_noise = abs(self._rng.normal(0.0, self.spread_std))
        spread_level = self.spread_mean * (1.0 + 0.35 * (1.0 - abs(self._signal)))
        self._spread = float(max(self.tick_size, spread_level + spread_noise))

        bid_mult = max(0.15, 1.0 + self.depth_sensitivity * self._signal)
        ask_mult = max(0.15, 1.0 - self.depth_sensitivity * self._signal)
        bid_noise = max(0.25, 1.0 + 0.12 * self._rng.standard_normal())
        ask_noise = max(0.25, 1.0 + 0.12 * self._rng.standard_normal())

        self._bid_vol_1 = float(max(1.0, self.base_depth * bid_mult * bid_noise))
        self._ask_vol_1 = float(max(1.0, self.base_depth * ask_mult * ask_noise))

    def _obi(self) -> float:
        denom = max(self._bid_vol_1 + self._ask_vol_1, 1.0)
        return float(np.clip(self._bid_vol_1 / denom, 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        inv_norm = self._inventory / max(1.0, float(self.total_inventory))
        time_norm = max(0.0, 1.0 - self._step / max(1.0, float(self.n_steps)))
        mid_norm = self._mid_price / max(self._benchmark_price, 1e-8)
        spread_norm = self._spread / max(self.spread_mean, 1e-8)
        bid_vol_norm = self._bid_vol_1 / max(self.base_depth, 1e-8)
        obi = self._obi()
        return np.array([inv_norm, time_norm, mid_norm, spread_norm, bid_vol_norm, obi], dtype=np.float32)

    def _info(
        self,
        exec_price: float,
        mid_price: float,
        executed_qty: int,
        forced_qty: int,
        step_is: float,
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
            "forced_qty": int(forced_qty),
            "step_is": float(step_is),
            "benchmark_price": float(self._benchmark_price),
            "spread": float(self._spread),
            "bid_vol_1": float(self._bid_vol_1),
            "order_book_imbalance": float(self._obi()),
            "completion_rate": sold / max(1.0, float(self.total_inventory)),
            "regime": float(self._regime),
            "signal": float(self._signal),
        }
