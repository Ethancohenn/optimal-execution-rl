"""Gymnasium execution environment backed by a stateful lightweight LOB."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from execution_infra.config import EnvConfig
from execution_infra.market_sim import LOBSnapshot, MarketSimulator


class ExecutionEnv(gym.Env):
    """Optimal-execution environment with endogenous LOB state transitions."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.cfg.n_actions)

        self.market = MarketSimulator(self.cfg)
        self.benchmark_price: float = 0.0
        self.inventory: int = 0
        self.current_step: int = 0
        self.cash: float = 0.0
        self.trades: list[dict] = []

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
        self.cash = 0.0
        self.trades = []
        return self._obs(lob), self._info(lob)

    def step_with_qty(self, quantity: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute an exact sell quantity, bypassing the discrete action space."""
        return self._execute_quantity(int(quantity))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        fraction = self.cfg.action_fractions[int(action)]
        quantity = int(np.round(fraction * self.inventory))
        return self._execute_quantity(quantity)

    def render(self) -> None:
        lob = self.market.lob
        print(
            f"Step {self.current_step:>3}/{self.cfg.n_steps} | "
            f"Inventory {self.inventory:>5} | "
            f"Cash {self.cash:>10.2f} | "
            f"Mid {lob.mid_price:>8.2f} | "
            f"Spread {lob.spread:>6.3f} | "
            f"IS {self._mark_to_market_shortfall(lob.mid_price):>9.2f}"
        )

    def _execute_quantity(self, quantity: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= self.cfg.n_steps:
            raise RuntimeError("Episode already ended. Call reset() before stepping again.")

        quantity = max(0, min(int(quantity), self.inventory))
        next_step = self.current_step + 1

        trade, lob = self.market.step(quantity)
        self.inventory -= trade.quantity
        self.cash += trade.cash_flow

        reward = trade.cash_flow - trade.quantity * trade.reference_mid
        step_is = (self.benchmark_price - trade.exec_price) * trade.quantity
        forced_qty = 0
        forced_exec_price = float("nan")
        forced_cash = 0.0
        terminal_penalty = 0.0

        if next_step >= self.cfg.n_steps and self.inventory > 0:
            if self.cfg.force_liquidation:
                forced_trade, lob = self.market.force_sell(self.inventory)
                forced_qty = forced_trade.quantity
                forced_exec_price = forced_trade.exec_price
                forced_cash = forced_trade.cash_flow
                self.inventory -= forced_qty
                self.cash += forced_cash
                reward += forced_cash - forced_qty * forced_trade.reference_mid
                step_is += (self.benchmark_price - forced_exec_price) * forced_qty
            else:
                terminal_penalty = self.cfg.lambda_penalty * self.inventory * lob.mid_price
                reward -= terminal_penalty

        if self.cfg.urgency_penalty > 0.0 and self.inventory > 0:
            reward -= (
                self.cfg.urgency_penalty
                * self._time_pressure()
                * (self.inventory / self.cfg.total_inventory)
            )

        executed_qty = trade.quantity + forced_qty
        combined_cash = trade.cash_flow + forced_cash
        exec_price = trade.reference_mid
        if executed_qty > 0:
            exec_price = combined_cash / executed_qty

        self.current_step = next_step
        terminated = self.inventory <= 0 and self.current_step < self.cfg.n_steps
        truncated = self.current_step >= self.cfg.n_steps

        if executed_qty > 0:
            self.trades.append(
                {
                    "step": self.current_step,
                    "quantity": executed_qty,
                    "exec_price": exec_price,
                    "mid_price": trade.reference_mid,
                    "next_mid_price": lob.mid_price,
                    "forced_qty": forced_qty,
                    "step_is": step_is,
                }
            )

        obs = self._obs(lob)
        info = self._info(
            lob,
            exec_price=exec_price,
            mid_price=trade.reference_mid,
            next_mid_price=lob.mid_price,
            executed_qty=executed_qty,
            forced_qty=forced_qty,
            forced_exec_price=forced_exec_price,
            step_is=step_is,
            terminal_penalty=terminal_penalty,
        )

        if self.render_mode == "human" and (terminated or truncated):
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _obs(self, lob: LOBSnapshot) -> np.ndarray:
        inv_norm = self.inventory / max(1, self.cfg.total_inventory)
        time_norm = max(0.0, 1.0 - self.current_step / max(1, self.cfg.n_steps))

        spread_ticks = lob.spread / max(self.cfg.tick_size, 1e-8)
        spread_span = max(1, self.cfg.max_spread_ticks - self.cfg.min_spread_ticks)
        spread_norm = np.clip((spread_ticks - self.cfg.min_spread_ticks) / spread_span, 0.0, 1.0)

        imbalance = float(lob.imbalance(self.cfg.obs_depth_levels))
        micro_signal = float(lob.microprice_signal)

        depth_cap = 2.0 * self.cfg.obs_depth_levels * (self.cfg.lob_depth_mean + 3 * self.cfg.lob_depth_std)
        depth_norm = np.clip(lob.top_depth(self.cfg.obs_depth_levels) / max(depth_cap, 1.0), 0.0, 1.0)

        return np.array(
            [inv_norm, time_norm, spread_norm, imbalance, micro_signal, depth_norm],
            dtype=np.float32,
        )

    def _info(
        self,
        lob: LOBSnapshot,
        *,
        exec_price: float = float("nan"),
        mid_price: float | None = None,
        next_mid_price: float | None = None,
        executed_qty: int = 0,
        forced_qty: int = 0,
        forced_exec_price: float = float("nan"),
        step_is: float = 0.0,
        terminal_penalty: float = 0.0,
    ) -> dict:
        current_mid = float(next_mid_price if next_mid_price is not None else lob.mid_price)
        decision_mid = float(mid_price if mid_price is not None else lob.mid_price)
        shares_sold = self.cfg.total_inventory - self.inventory
        mtm_shortfall = self._mark_to_market_shortfall(current_mid)

        return {
            "inventory_remaining": self.inventory,
            "remaining_inventory": self.inventory,
            "shares_sold": shares_sold,
            "step": self.current_step,
            "t": self.current_step,
            "exec_price": float(exec_price),
            "forced_exec_price": float(forced_exec_price),
            "mid_price": decision_mid,
            "next_mid_price": current_mid,
            "spread": float(lob.spread),
            "imbalance": float(lob.imbalance(self.cfg.obs_depth_levels)),
            "top_depth": float(lob.top_depth(self.cfg.obs_depth_levels)),
            "executed_qty": int(executed_qty),
            "forced_qty": int(forced_qty),
            "cash": float(self.cash),
            "step_is": float(step_is),
            "benchmark_price": float(self.benchmark_price),
            "completion_rate": shares_sold / max(1, self.cfg.total_inventory),
            "implementation_shortfall": float(mtm_shortfall),
            "total_cost": float(mtm_shortfall),
            "terminal_penalty": float(terminal_penalty),
        }

    def _mark_to_market_shortfall(self, current_mid: float) -> float:
        benchmark_notional = self.benchmark_price * self.cfg.total_inventory
        portfolio_value = self.cash + self.inventory * current_mid
        return benchmark_notional - portfolio_value

    def _time_pressure(self) -> float:
        return min(1.0, self.current_step / max(1, self.cfg.n_steps))
