"""
Gymnasium environment for the optimal-execution task.

The agent must liquidate `total_inventory` shares over `n_steps`
time-steps while minimising execution cost (implementation shortfall).

Observation  (Box, 4-dim float32)
─────────────────────────────────
  0  inventory_remaining   normalised  [0, 1]
  1  time_remaining        normalised  [1 → 0]
  2  spread                raw dollars
  3  lob_volume            normalised to [0, 1]  (sum of top-3 bid+ask / max_vol)

Action  (Discrete, n_actions)
─────────────────────────────
  Maps to selling a fraction of *remaining* inventory:
      action 0 → 0 %   (wait)
      action 1 → 25 %
      action 2 → 50 %
      action 3 → 75 %
      action 4 → 100 %

Reward
──────
  r_t = -(P_executed - P_benchmark) · q_t

  At the terminal step an additional penalty is applied if
  inventory is not fully liquidated:
      r_T -= λ · leftover_inventory · P_benchmark
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from execution_infra.config import EnvConfig
from execution_infra.market_sim import MarketSimulator


class ExecutionEnv(gym.Env):
    """
    Optimal-execution Gymnasium environment.

    Parameters
    ----------
    config : EnvConfig | None
        If *None*, default hyper-parameters are used.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        # ── spaces ────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.cfg.n_actions)

        # ── internal state ────────────────────────────────────────
        self.market = MarketSimulator(self.cfg)
        self.benchmark_price: float = 0.0
        self.inventory: int = 0
        self.current_step: int = 0

        # episode tracking (exposed in info dict)
        self.total_cost: float = 0.0
        self.total_shares_sold: int = 0
        self.trades: list[dict] = []

    # ── gymnasium API ─────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.market.rng = np.random.default_rng(seed)

        lob = self.market.reset()
        self.benchmark_price = lob.mid_price
        self.inventory = self.cfg.total_inventory
        self.current_step = 0
        self.total_cost = 0.0
        self.total_shares_sold = 0
        self.trades = []

        return self._obs(lob), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time-step.

        Returns (obs, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # ── 1. map action → quantity ──────────────────────────────
        frac = self.cfg.action_fractions[action]
        quantity = int(np.round(frac * self.inventory))
        # Ensure any non-wait action actually trades at least one share.
        if action > 0 and self.inventory > 0 and quantity == 0:
            quantity = 1
        # At the deadline, any non-wait action liquidates all remaining shares.
        if action > 0 and self.inventory > 0 and self.current_step == (self.cfg.n_steps - 1):
            quantity = self.inventory
        # Clamp to available inventory
        quantity = min(quantity, self.inventory)

        # ── 2. execute trade in simulator ─────────────────────────
        exec_price, lob = self.market.step(quantity)

        # ── 3. compute step reward ────────────────────────────────
        #  r_t = -(P_exec - P_bench) · q_t
        #  (selling below benchmark → negative shortfall → positive r)
        cost = (exec_price - self.benchmark_price) * quantity
        reward = -cost

        # ── 4. update bookkeeping ─────────────────────────────────
        self.inventory -= quantity
        self.current_step += 1
        self.total_cost += cost
        self.total_shares_sold += quantity

        if quantity > 0:
            self.trades.append({
                "step": self.current_step,
                "quantity": quantity,
                "exec_price": exec_price,
                "mid_price": lob.mid_price,
                "cost": cost,
            })

        # ── 5. check termination ──────────────────────────────────
        terminated = (self.inventory <= 0)
        truncated = (self.current_step >= self.cfg.n_steps) and (self.inventory > 0)

        # Terminal inventory penalty
        if truncated:
            penalty = self.cfg.lambda_penalty * self.inventory * self.benchmark_price
            reward -= penalty
            self.total_cost += penalty

        done = terminated or truncated
        obs = self._obs(lob)
        info = self._info()

        if self.render_mode == "human" and done:
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable summary of the current episode state."""
        pct_done = self.total_shares_sold / self.cfg.total_inventory * 100
        avg_price = (
            self.total_cost / self.total_shares_sold
            if self.total_shares_sold > 0
            else 0.0
        )
        print(
            f"Step {self.current_step:>3}/{self.cfg.n_steps} | "
            f"Inventory {self.inventory:>6} ({pct_done:5.1f}% sold) | "
            f"Avg cost/share {avg_price:+.4f} | "
            f"Benchmark {self.benchmark_price:.2f} | "
            f"Mid {self.market.mid_price:.2f}"
        )

    # ── private helpers ───────────────────────────────────────────

    def _obs(self, lob) -> np.ndarray:
        """Build the 4-dim observation vector."""
        inv_norm = self.inventory / self.cfg.total_inventory
        time_norm = 1.0 - self.current_step / self.cfg.n_steps
        spread = lob.spread
        # Normalise LOB volume:  total_vol / (n_levels * 2 * depth_mean)
        max_vol = self.cfg.n_lob_levels * 2 * self.cfg.lob_depth_mean
        vol_norm = lob.total_volume / max_vol if max_vol > 0 else 0.0
        return np.array([inv_norm, time_norm, spread, vol_norm], dtype=np.float32)

    def _info(self) -> dict:
        """Return episode metrics dict."""
        return {
            "inventory_remaining": self.inventory,
            "shares_sold": self.total_shares_sold,
            "total_cost": self.total_cost,
            "implementation_shortfall": self.total_cost,
            "benchmark_price": self.benchmark_price,
            "step": self.current_step,
            "completion_rate": self.total_shares_sold / self.cfg.total_inventory,
        }
