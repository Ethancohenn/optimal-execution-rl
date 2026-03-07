from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.common.actions import DEFAULT_ACTION_SPEC, action_to_qty

class StubExecutionEnv(gym.Env):
    '''A minimal Gymnasium env for testing the RL pipeline.

    Observation:
      [inv_remaining_norm, time_remaining_norm, mid_price_norm, spread_norm, vol_top1_norm, vol_top3_norm]
    Action:
      Discrete index into fractions list.
    '''

    metadata = {"render_modes": []}

    def __init__(
        self,
        T: int = 20,
        Q0: float = 1000.0,
        max_trade_size: float | None = None,
        seed: int = 0,
        force_liquidation: bool = True,
        impact_coeff: float = 0.0002,
        terminal_impact_coeff: float = 0.0010,
    ):
        super().__init__()
        self.T = int(T)
        self.Q0 = float(Q0)
        self.max_trade_size = float(max_trade_size) if max_trade_size is not None else (self.Q0 / self.T)
        self.force_liquidation = bool(force_liquidation)
        self.impact_coeff = float(impact_coeff)
        self.terminal_impact_coeff = float(terminal_impact_coeff)
        self.spec_actions = DEFAULT_ACTION_SPEC

        self.action_space = spaces.Discrete(len(self.spec_actions.fractions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.remaining_inventory = self.Q0

        # synthetic market state
        self.mid_price = 100.0
        self.spread = 0.02
        self.vol_top1 = 500.0
        self.vol_top3 = 1500.0

        obs = self._get_obs()
        info = self._get_info(executed_qty=0.0, exec_price=np.nan)
        return obs, info

    def step(self, action: int):
        # price evolves slightly
        noise = float(self._rng.normal(0.0, 0.01))
        self.mid_price = float(self.mid_price + noise)

        # execute qty based on action
        action_qty = action_to_qty(action, self.remaining_inventory, self.max_trade_size)
        action_is = self.impact_coeff * (action_qty ** 2)
        executed_qty = float(action_qty)
        forced_qty = 0.0

        self.remaining_inventory = float(self.remaining_inventory - action_qty)

        self.t += 1
        terminated = (self.t >= self.T)
        truncated = False

        forced_is = 0.0
        if terminated and self.remaining_inventory > 1e-6 and self.force_liquidation:
            # Force liquidation at episode end so policies cannot avoid cost by not finishing.
            forced_qty = float(self.remaining_inventory)
            forced_is = self.terminal_impact_coeff * (forced_qty ** 2)
            executed_qty += forced_qty
            self.remaining_inventory = 0.0

        # Equivalent average execution price for this step.
        step_is = float(action_is + forced_is)
        exec_price = float(self.mid_price - (step_is / executed_qty)) if executed_qty > 1e-12 else float("nan")
        reward = -step_is

        # Legacy behavior for experiments that disable forced liquidation.
        if terminated and self.remaining_inventory > 1e-6 and not self.force_liquidation:
            reward -= float(0.01 * self.remaining_inventory)

        obs = self._get_obs()
        info = self._get_info(executed_qty=executed_qty, exec_price=exec_price, forced_qty=forced_qty, step_is=step_is)
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        inv_norm = self.remaining_inventory / self.Q0
        time_norm = (self.T - self.t) / self.T
        mid_norm = self.mid_price / 100.0
        spread_norm = self.spread / 0.02
        vol1_norm = self.vol_top1 / 500.0
        vol3_norm = self.vol_top3 / 1500.0
        return np.array([inv_norm, time_norm, mid_norm, spread_norm, vol1_norm, vol3_norm], dtype=np.float32)

    def _get_info(self, executed_qty: float, exec_price: float, forced_qty: float = 0.0, step_is: float = 0.0):
        return {
            "executed_qty": float(executed_qty),
            "exec_price": float(exec_price) if not np.isnan(exec_price) else float("nan"),
            "mid_price": float(self.mid_price),
            "spread": float(self.spread),
            "remaining_inventory": float(self.remaining_inventory),
            "forced_qty": float(forced_qty),
            "step_is": float(step_is),
            "t": int(self.t),
        }
