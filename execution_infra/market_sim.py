"""Stateful lightweight limit-order-book simulator for execution RL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution_infra.config import EnvConfig


@dataclass
class LOBSnapshot:
    """Compact representation of the current order-book state."""

    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    bid_volumes: np.ndarray
    ask_volumes: np.ndarray
    fundamental_price: float
    recent_order_flow: float

    @property
    def total_volume(self) -> float:
        return float(self.bid_volumes.sum() + self.ask_volumes.sum())

    def top_depth(self, levels: int = 3) -> float:
        k = max(1, min(int(levels), len(self.bid_volumes), len(self.ask_volumes)))
        return float(self.bid_volumes[:k].sum() + self.ask_volumes[:k].sum())

    def imbalance(self, levels: int = 3) -> float:
        k = max(1, min(int(levels), len(self.bid_volumes), len(self.ask_volumes)))
        bid_depth = float(self.bid_volumes[:k].sum())
        ask_depth = float(self.ask_volumes[:k].sum())
        denom = bid_depth + ask_depth
        if denom <= 0.0:
            return 0.0
        return (bid_depth - ask_depth) / denom

    @property
    def microprice(self) -> float:
        top_bid = float(self.bid_volumes[0])
        top_ask = float(self.ask_volumes[0])
        denom = top_bid + top_ask
        if denom <= 0.0:
            return self.mid_price
        return (self.best_ask * top_bid + self.best_bid * top_ask) / denom

    @property
    def microprice_signal(self) -> float:
        if self.spread <= 0.0:
            return 0.0
        signal = 2.0 * (self.microprice - self.mid_price) / self.spread
        return float(np.clip(signal, -1.0, 1.0))


@dataclass
class ExecutionResult:
    quantity: int
    exec_price: float
    cash_flow: float
    reference_mid: float
    next_mid: float


class MarketSimulator:
    """Stateful multi-level LOB with endogenous execution impact."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.best_bid_tick: int = 0
        self.best_ask_tick: int = 0
        self.bid_volumes = np.zeros(self.cfg.n_lob_levels, dtype=np.int64)
        self.ask_volumes = np.zeros(self.cfg.n_lob_levels, dtype=np.int64)
        self.fundamental_price: float = 0.0
        self.order_flow_ema: float = 0.0
        self.last_agent_qty: int = 0
        self.cumulative_agent_volume: int = 0
        self._liquidity_scale: float = 1.0

    def reset(self) -> LOBSnapshot:
        """Reset the book around the initial reference price."""
        center_tick = self._price_to_tick(self.cfg.initial_price)
        spread_ticks = self._sample_spread_ticks()
        bid_offset = spread_ticks // 2

        self.best_bid_tick = center_tick - bid_offset
        self.best_ask_tick = self.best_bid_tick + spread_ticks
        self.fundamental_price = self.cfg.initial_price
        self.order_flow_ema = 0.0
        self.last_agent_qty = 0
        self.cumulative_agent_volume = 0
        self._liquidity_scale = float(np.clip(self.rng.lognormal(mean=0.0, sigma=0.18), 0.7, 1.4))

        self.bid_volumes = np.array(
            [self._sample_level_volume(level) for level in range(self.cfg.n_lob_levels)],
            dtype=np.int64,
        )
        self.ask_volumes = np.array(
            [self._sample_level_volume(level) for level in range(self.cfg.n_lob_levels)],
            dtype=np.int64,
        )
        self._normalize_book()
        return self.lob

    def step(self, quantity: int) -> tuple[ExecutionResult, LOBSnapshot]:
        """Execute an agent sell order and advance the book by one decision step."""
        quantity = max(0, int(quantity))
        reference_mid = self.mid_price
        executed_qty, exec_price = self._execute_sell(quantity)

        self.last_agent_qty = executed_qty
        self.cumulative_agent_volume += executed_qty
        self._advance_market(agent_qty=executed_qty)
        snapshot = self.lob
        result = ExecutionResult(
            quantity=executed_qty,
            exec_price=exec_price,
            cash_flow=float(executed_qty * exec_price),
            reference_mid=reference_mid,
            next_mid=snapshot.mid_price,
        )
        return result, snapshot

    def force_sell(self, quantity: int) -> tuple[ExecutionResult, LOBSnapshot]:
        """Execute a deadline liquidation without simulating another exogenous step."""
        quantity = max(0, int(quantity))
        reference_mid = self.mid_price
        executed_qty, exec_price = self._execute_sell(quantity)
        self.last_agent_qty = executed_qty
        self.cumulative_agent_volume += executed_qty
        self._update_flow_memory(agent_qty=executed_qty, external_buy_qty=0, external_sell_qty=0)
        self._update_fundamental(agent_qty=executed_qty)
        snapshot = self.lob
        result = ExecutionResult(
            quantity=executed_qty,
            exec_price=exec_price,
            cash_flow=float(executed_qty * exec_price),
            reference_mid=reference_mid,
            next_mid=snapshot.mid_price,
        )
        return result, snapshot

    @property
    def mid_price(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def best_bid(self) -> float:
        return self.best_bid_tick * self.cfg.tick_size

    @property
    def best_ask(self) -> float:
        return self.best_ask_tick * self.cfg.tick_size

    @property
    def lob(self) -> LOBSnapshot:
        return LOBSnapshot(
            mid_price=self.mid_price,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            spread=self.best_ask - self.best_bid,
            bid_volumes=self.bid_volumes.astype(np.float32, copy=True),
            ask_volumes=self.ask_volumes.astype(np.float32, copy=True),
            fundamental_price=float(self.fundamental_price),
            recent_order_flow=float(self.order_flow_ema),
        )

    def _advance_market(self, agent_qty: int) -> None:
        snapshot = self.lob
        imbalance = snapshot.imbalance(self.cfg.obs_depth_levels)
        spread = max(snapshot.spread, self.cfg.tick_size)
        mispricing = float(np.clip((self.fundamental_price - snapshot.mid_price) / spread, -2.0, 2.0))

        buy_intensity = self.cfg.market_order_intensity * np.exp(
            0.40 * imbalance + 0.30 * mispricing + 0.20 * self.order_flow_ema
        )
        sell_intensity = self.cfg.market_order_intensity * np.exp(
            -0.40 * imbalance - 0.30 * mispricing - 0.20 * self.order_flow_ema
        )
        external_buy_qty = self._sample_flow_qty(buy_intensity)
        external_sell_qty = self._sample_flow_qty(sell_intensity)

        if external_buy_qty > 0:
            self._consume_ask_liquidity(external_buy_qty)
        if external_sell_qty > 0:
            self._consume_bid_liquidity(external_sell_qty)

        self._update_flow_memory(
            agent_qty=agent_qty,
            external_buy_qty=external_buy_qty,
            external_sell_qty=external_sell_qty,
        )
        self._apply_cancellations()
        self._apply_limit_arrivals(mispricing=mispricing)
        self._apply_quote_improvements(mispricing=mispricing)
        self._update_fundamental(agent_qty=agent_qty)
        self._normalize_book()

    def _execute_sell(self, quantity: int) -> tuple[int, float]:
        if quantity <= 0:
            return 0, self.mid_price
        return self._consume_bid_liquidity(quantity)

    def _consume_bid_liquidity(self, quantity: int) -> tuple[int, float]:
        return self._consume_side(quantity=quantity, side="bid")

    def _consume_ask_liquidity(self, quantity: int) -> tuple[int, float]:
        return self._consume_side(quantity=quantity, side="ask")

    def _consume_side(self, quantity: int, side: str) -> tuple[int, float]:
        if quantity <= 0:
            return 0, self.mid_price

        executed = 0
        notional = 0.0

        while executed < quantity:
            remaining = quantity - executed
            if side == "bid":
                available = int(self.bid_volumes[0])
                trade_qty = min(remaining, available)
                notional += trade_qty * self.best_bid
                self.bid_volumes[0] -= trade_qty
                executed += trade_qty
                while self.bid_volumes[0] <= 0:
                    self.best_bid_tick -= 1
                    self.bid_volumes = np.concatenate(
                        [self.bid_volumes[1:], np.array([self._sample_tail_volume()], dtype=np.int64)]
                    )
            else:
                available = int(self.ask_volumes[0])
                trade_qty = min(remaining, available)
                notional += trade_qty * self.best_ask
                self.ask_volumes[0] -= trade_qty
                executed += trade_qty
                while self.ask_volumes[0] <= 0:
                    self.best_ask_tick += 1
                    self.ask_volumes = np.concatenate(
                        [self.ask_volumes[1:], np.array([self._sample_tail_volume()], dtype=np.int64)]
                    )

        exec_price = notional / max(1, executed)
        self._normalize_book()
        return executed, float(exec_price)

    def _apply_limit_arrivals(self, mispricing: float) -> None:
        buy_bias = float(np.clip(1.0 + 0.25 * mispricing + 0.35 * self.order_flow_ema, 0.6, 1.8))
        sell_bias = float(np.clip(1.0 - 0.25 * mispricing - 0.35 * self.order_flow_ema, 0.6, 1.8))

        for level in range(self.cfg.n_lob_levels):
            level_scale = np.exp(-0.50 * level)
            bid_qty = self._sample_flow_qty(self.cfg.limit_order_intensity * level_scale * buy_bias)
            ask_qty = self._sample_flow_qty(self.cfg.limit_order_intensity * level_scale * sell_bias)
            self.bid_volumes[level] += bid_qty
            self.ask_volumes[level] += ask_qty

    def _apply_quote_improvements(self, mispricing: float) -> None:
        spread_ticks = self.best_ask_tick - self.best_bid_tick
        if spread_ticks <= 1:
            return

        bid_improve_prob = self.cfg.quote_improve_prob * float(
            np.clip(0.6 + 0.5 * self.order_flow_ema + 0.25 * mispricing, 0.0, 1.0)
        )
        ask_improve_prob = self.cfg.quote_improve_prob * float(
            np.clip(0.6 - 0.5 * self.order_flow_ema - 0.25 * mispricing, 0.0, 1.0)
        )

        if self.rng.random() < bid_improve_prob:
            self.best_bid_tick += 1
            self.bid_volumes = np.concatenate(
                [np.array([self._sample_level_volume(0)], dtype=np.int64), self.bid_volumes[:-1]]
            )

        if (self.best_ask_tick - self.best_bid_tick) > 1 and self.rng.random() < ask_improve_prob:
            self.best_ask_tick -= 1
            self.ask_volumes = np.concatenate(
                [np.array([self._sample_level_volume(0)], dtype=np.int64), self.ask_volumes[:-1]]
            )

    def _apply_cancellations(self) -> None:
        for level in range(self.cfg.n_lob_levels):
            base_prob = float(np.clip(self.cfg.cancellation_rate * (1.0 + 0.15 * level), 0.0, 0.95))
            bid_prob = float(np.clip(base_prob * (1.0 + 0.6 * max(-self.order_flow_ema, 0.0)), 0.0, 0.95))
            ask_prob = float(np.clip(base_prob * (1.0 + 0.6 * max(self.order_flow_ema, 0.0)), 0.0, 0.95))
            self.bid_volumes[level] -= self.rng.binomial(int(self.bid_volumes[level]), bid_prob)
            self.ask_volumes[level] -= self.rng.binomial(int(self.ask_volumes[level]), ask_prob)

    def _update_flow_memory(self, agent_qty: int, external_buy_qty: int, external_sell_qty: int) -> None:
        depth_scale = max(1.0, self.lob.top_depth(self.cfg.obs_depth_levels))
        signed_flow = (external_buy_qty - external_sell_qty - agent_qty) / depth_scale
        memory = float(np.clip(self.cfg.order_flow_memory, 0.0, 1.0))
        self.order_flow_ema = (1.0 - memory) * self.order_flow_ema + memory * signed_flow

    def _update_fundamental(self, agent_qty: int) -> None:
        anchor_price = self.mid_price
        reversion = self.cfg.fundamental_mean_reversion * (anchor_price - self.fundamental_price)
        flow_push = 0.75 * self.order_flow_ema * self.cfg.tick_size
        impact = (
            self.cfg.agent_permanent_impact
            * agent_qty
            / max(1.0, self.cfg.obs_depth_levels * self.cfg.lob_depth_mean)
            * self.cfg.tick_size
        )
        noise = float(self.rng.normal(0.0, self.cfg.volatility))
        self.fundamental_price += reversion + flow_push - impact + noise

    def _normalize_book(self) -> None:
        self.bid_volumes = np.maximum(self.bid_volumes.astype(np.int64, copy=False), 0)
        self.ask_volumes = np.maximum(self.ask_volumes.astype(np.int64, copy=False), 0)

        while self.bid_volumes[0] <= 0:
            self.best_bid_tick -= 1
            self.bid_volumes = np.concatenate(
                [self.bid_volumes[1:], np.array([self._sample_tail_volume()], dtype=np.int64)]
            )
        while self.ask_volumes[0] <= 0:
            self.best_ask_tick += 1
            self.ask_volumes = np.concatenate(
                [self.ask_volumes[1:], np.array([self._sample_tail_volume()], dtype=np.int64)]
            )
        if self.best_ask_tick <= self.best_bid_tick:
            self.best_ask_tick = self.best_bid_tick + 1
        self.bid_volumes = np.maximum(self.bid_volumes, 0)
        self.ask_volumes = np.maximum(self.ask_volumes, 0)

    def _sample_spread_ticks(self) -> int:
        spread = max(self.cfg.tick_size, float(self.rng.normal(self.cfg.spread_mean, self.cfg.spread_std)))
        spread_ticks = max(self.cfg.min_spread_ticks, int(round(spread / self.cfg.tick_size)))
        return min(spread_ticks, self.cfg.max_spread_ticks)

    def _sample_level_volume(self, level: int) -> int:
        mean = self.cfg.lob_depth_mean * np.exp(-self.cfg.depth_decay * level) * self._liquidity_scale
        std = max(1.0, self.cfg.lob_depth_std * np.exp(-0.5 * self.cfg.depth_decay * level))
        draw = self.rng.normal(mean, std)
        return int(max(1, round(draw)))

    def _sample_tail_volume(self) -> int:
        return self._sample_level_volume(self.cfg.n_lob_levels - 1)

    def _sample_flow_qty(self, intensity: float) -> int:
        lots = int(self.rng.poisson(max(0.0, intensity)))
        return lots * self.cfg.market_lot_size

    def _price_to_tick(self, price: float) -> int:
        return int(round(price / self.cfg.tick_size))
