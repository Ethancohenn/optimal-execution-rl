"""
Gymnasium environment for optimal execution that replays ABIDES data.

Instead of generating synthetic market dynamics, this env loads a pre-extracted
feature matrix from a .npz file (produced by execution_infra.feature_extraction.pipeline)
and replays contiguous windows of rows as the market backdrop.

Episode = a randomly sampled contiguous window of `n_steps` rows.
The agent's execution decisions apply on top of real ABIDES mid-prices with
the same market-impact formula used in ExecutionEnv.

Observation (6-dim float32)
───────────────────────────
  0  inv_norm          remaining inventory / total_inventory    [0, 1]
  1  time_norm         steps left / n_steps                     [1 → 0]
  2  mid_price_norm    mid_price / benchmark_price              [≈1]
  3  spread_norm       spread / mean_spread                     [>0]
  4  bid_vol_norm      bid_vol_1 / mean_bid_vol                 [>0]
  5  obi               order_book_imbalance                     [0, 1]

Action (Discrete, n_actions)
─────────────────────────────
  Same as ExecutionEnv: fraction of remaining inventory to sell.
  [0.0, 0.25, 0.50, 0.75, 1.0] for n_actions=5.

Reward
──────
  r_t = (exec_price - benchmark_price) * qty
  Terminal penalty for leftover: -lambda_penalty * leftover * benchmark_price
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AbidesReplayEnv(gym.Env):
    """Optimal-execution env backed by replayed ABIDES feature data.

    Parameters
    ----------
    npz_path : str
        Path to the .npz produced by execution_infra.feature_extraction.pipeline.
    n_steps : int
        Number of decision steps per episode (window length).
    total_inventory : int
        Total shares the agent must liquidate.
    n_actions : int
        Number of discrete sell-fraction actions (default 5).
    eta : float
        Temporary market-impact coefficient.
    gamma_perm : float
        Permanent market-impact coefficient (applied to exec price).
    lambda_penalty : float
        Weight for the terminal leftover-inventory penalty.
    """

    metadata = {"render_modes": ["human"]}

    # Columns we read from the npz (must exist in features.npz)
    _REQUIRED_COLS = ["mid_price", "spread", "bid_vol_1", "order_book_imbalance"]

    def __init__(
        self,
        npz_path: str = "data/features.npz",
        n_steps: int = 60,
        total_inventory: int = 1_000,
        n_actions: int = 5,
        eta: float = 2.5e-6,
        gamma_perm: float = 2.5e-7,
        lambda_penalty: float = 1.0,
        urgency_coef: float = 0.0,
        render_mode: Optional[str] = None,
        norm_stats: Optional[dict] = None,
        split: str = "all",
    ) -> None:
        """Parameters
        ----------
        urgency_coef : float
            Per-step holding penalty coefficient.  At step t the agent pays
            ``urgency_coef * (t / n_steps) * (inventory / total_inventory)``
            extra cost, growing linearly toward the deadline.  This teaches
            the Q-function to sell progressively without forced liquidation.
            A value of ~50 works well with the default IS scale.
        norm_stats : dict, optional
            Override normalisation stats with pre-computed values from a
            training dataset. Keys: ``mean_spread``, ``mean_bid_vol``.
        """
        super().__init__()
        self.n_steps = int(n_steps)
        self.total_inventory = int(total_inventory)
        self.n_actions = int(n_actions)
        self.eta = float(eta)
        self.gamma_perm = float(gamma_perm)
        self.lambda_penalty = float(lambda_penalty)
        self.urgency_coef = float(urgency_coef)
        self.render_mode = render_mode

        # ── Load feature matrix ───────────────────────────────────
        npz = np.load(str(npz_path))
        self._features: np.ndarray = npz["features"].astype(np.float32)  # (T, F)
        self._col_names: list[str] = list(npz["feature_names"])
        
        # ── Train/Test Split (80/20) ──────────────────────────────
        T_full = self._features.shape[0]
        split_idx = int(T_full * 0.8)
        if split == "train":
            self._features = self._features[:split_idx]
        elif split == "test":
            self._features = self._features[split_idx:]
        elif split != "all":
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', or 'all'.")

        T, F = self._features.shape

        if T < self.n_steps + 1:
            raise ValueError(
                f"features.npz has only {T} rows but n_steps={self.n_steps}. "
                "Use a smaller n_steps or a longer simulation."
            )

        missing = [c for c in self._REQUIRED_COLS if c not in self._col_names]
        if missing:
            raise ValueError(f"Missing columns in features.npz: {missing}")

        # Build column index lookup
        self._ci: dict[str, int] = {name: i for i, name in enumerate(self._col_names)}

        # Normalisation statistics — use provided ones (for eval on held-out data)
        # or compute from the loaded dataset (for training).
        if norm_stats is not None:
            self._mean_spread  = float(norm_stats["mean_spread"])
            self._mean_bid_vol = float(norm_stats["mean_bid_vol"])
        else:
            self._mean_spread  = float(np.nanmean(self._features[:, self._ci["spread"]]))
            self._mean_bid_vol = float(np.nanmean(self._features[:, self._ci["bid_vol_1"]]))
        self._mean_spread  = max(self._mean_spread,  1e-8)
        self._mean_bid_vol = max(self._mean_bid_vol, 1.0)

        self._T = T  # total rows available

        # ── Action fractions ──────────────────────────────────────
        if self.n_actions == 1:
            self._fractions = [1.0]
        else:
            self._fractions = [i / (self.n_actions - 1) for i in range(self.n_actions)]

        # ── Gymnasium spaces ──────────────────────────────────────
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Episode state (set in reset) ──────────────────────────
        self._start: int = 0
        self._step: int = 0
        self._inventory: int = 0
        self._benchmark_price: float = 0.0
        self._cumulative_volume: float = 0.0
        self.trades: list[dict] = []

        self._rng = np.random.default_rng(None)

    # ── Gymnasium API ──────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a random starting row so the window fits
        max_start = self._T - self.n_steps - 1
        self._start = int(self._rng.integers(0, max_start + 1))
        self._step = 0
        self._inventory = self.total_inventory
        self._cumulative_volume = 0.0
        self.trades = []

        # Benchmark = mid-price at first row of the window (arrival price)
        self._benchmark_price = float(self._row(0)["mid_price"])

        return self._obs(), self._info()

    def step_with_qty(self, quantity: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute exactly `quantity` shares, bypassing the discrete action space.

        Useful for baselines (e.g. TWAP) that need to sell a fixed number of
        shares per step regardless of the action fraction granularity.
        """
        quantity = max(0, min(int(quantity), self._inventory))
        row = self._row(self._step)
        mid = row["mid_price"]

        if quantity == 0:
            exec_price = mid
        else:
            impact = self.eta * quantity + self.gamma_perm * self._cumulative_volume
            exec_price = mid * (1.0 - impact)
            exec_price = max(exec_price, 1e-4)

        #Reward is the negative implementation shortfall contribution relative
        #to the arrival benchmark price.
        reward = (exec_price - self._benchmark_price) * quantity

        # Urgency shaping: penalise holding inventory near deadline.
        time_pressure = self._step / self.n_steps
        inv_fraction  = self._inventory / self.total_inventory
        reward -= self.urgency_coef * time_pressure * inv_fraction

        self._cumulative_volume += quantity
        self._inventory -= quantity
        self._step += 1

        if quantity > 0:
            self.trades.append({
                "step": self._step,
                "quantity": quantity,
                "exec_price": exec_price,
                "mid_price": mid,
                "is_contrib": (self._benchmark_price - exec_price) * quantity,
                "reward": reward,
            })

        terminated = self._inventory <= 0
        truncated = (self._step >= self.n_steps) and (self._inventory > 0)
        if truncated:
            penalty = self.lambda_penalty * self._inventory * self._benchmark_price
            reward -= penalty

        obs = self._obs()
        info = self._info(exec_price=exec_price, mid_price=mid, executed_qty=quantity)
        return obs, float(reward), terminated, truncated, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        row = self._row(self._step)
        mid = row["mid_price"]

        # ── 1. Map action → quantity ──────────────────────────────
        frac = self._fractions[action]
        quantity = int(np.round(frac * self._inventory))
        # Ensure any non-wait action actually trades at least one share.
        if action > 0 and self._inventory > 0 and quantity == 0:
            quantity = 1
        # At the deadline, any non-wait action liquidates all remaining shares.
        if action > 0 and self._inventory > 0 and self._step == (self.n_steps - 1):
            quantity = self._inventory
        quantity = min(quantity, self._inventory)

        # ── 2. Execution price (real mid + market impact) ─────────
        if quantity == 0:
            exec_price = mid
        else:
            impact = self.eta * quantity + self.gamma_perm * self._cumulative_volume
            exec_price = mid * (1.0 - impact)
            exec_price = max(exec_price, 1e-4)

        # ── 3. Reward ─────────────────────────────────────────────
        # Reward = negative implementation shortfall contribution relative to
        # the arrival benchmark price.
        reward = (exec_price - self._benchmark_price) * quantity

        # Urgency shaping: penalise holding inventory near deadline.
        time_pressure = self._step / self.n_steps
        inv_fraction  = self._inventory / self.total_inventory
        reward -= self.urgency_coef * time_pressure * inv_fraction

        # ── 4. Bookkeeping ────────────────────────────────────────
        self._cumulative_volume += quantity
        self._inventory -= quantity
        self._step += 1

        if quantity > 0:
            self.trades.append({
                "step": self._step,
                "quantity": quantity,
                "exec_price": exec_price,
                "mid_price": mid,
                "is_contrib": (self._benchmark_price - exec_price) * quantity,
                "reward": reward,
            })

        # ── 5. Termination ────────────────────────────────────────
        terminated = self._inventory <= 0
        truncated = (self._step >= self.n_steps) and (self._inventory > 0)

        if truncated:
            penalty = self.lambda_penalty * self._inventory * self._benchmark_price
            reward -= penalty

        obs = self._obs()
        info = self._info(exec_price=exec_price, mid_price=mid, executed_qty=quantity)

        if self.render_mode == "human" and (terminated or truncated):
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        sold = self.total_inventory - self._inventory
        pct = sold / self.total_inventory * 100
        print(
            f"Step {self._step:>3}/{self.n_steps} | "
            f"Inventory {self._inventory:>6} ({pct:5.1f}% sold) | "
            f"Benchmark {self._benchmark_price:.4f}"
        )

    # ── Private helpers ────────────────────────────────────────────

    def _row(self, offset: int) -> dict[str, float]:
        """Return a dict of column_name → value for the current episode row."""
        row = self._features[self._start + offset]
        return {name: float(row[i]) for name, i in self._ci.items()}

    def _obs(self) -> np.ndarray:
        offset = min(self._step, self.n_steps - 1)
        row = self._row(offset)

        inv_norm = self._inventory / self.total_inventory
        time_norm = 1.0 - self._step / self.n_steps
        mid_norm = row["mid_price"] / max(self._benchmark_price, 1e-8)
        # Clip spread and volume to prevent wild out-of-distribution 
        # observations during test evaluation causing the Q-network to break.
        spread_norm = float(np.clip(row["spread"] / self._mean_spread, 0.0, 10.0))
        bid_vol_norm = float(np.clip(row["bid_vol_1"] / self._mean_bid_vol, 0.0, 10.0))
        obi = float(np.clip(row["order_book_imbalance"], 0.0, 1.0))

        return np.array(
            [inv_norm, time_norm, mid_norm, spread_norm, bid_vol_norm, obi],
            dtype=np.float32,
        )

    def _info(
        self,
        exec_price: float = float("nan"),
        mid_price: float = float("nan"),
        executed_qty: int = 0,
    ) -> dict:
        return {
            "inventory_remaining": self._inventory,
            "shares_sold": self.total_inventory - self._inventory,
            "step": self._step,
            "exec_price": exec_price,
            "mid_price": mid_price,
            "executed_qty": executed_qty,
            "benchmark_price": self._benchmark_price,
            "completion_rate": (self.total_inventory - self._inventory) / self.total_inventory,
        }
