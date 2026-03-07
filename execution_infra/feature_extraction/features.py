"""
Feature engineering functions for ABIDES market data.

Each function takes a parsed DataFrame (from ``parsers.py``) and returns
a new DataFrame of derived features, all with a DatetimeIndex.

Feature groups
--------------
* **Orderbook** – best bid/ask, spread, mid-price, depth (level 1), imbalance
* **Exchange**  – last trade price, traded volume, trade intensity
* **Execution** – remaining inventory, executed volume, time remaining
* **Derived**   – volatility, VWAP, benchmark (fundamental) price

.. note::

    All **prices** in the output are in **dollars** (converted from ABIDES
    internal cents).  All **volumes** are in shares (unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Orderbook features  (from BEST_BID / BEST_ASK events)
# ---------------------------------------------------------------------------

def compute_orderbook_features(bba_df: pd.DataFrame) -> pd.DataFrame:
    """Compute order-book features from parsed best-bid/ask data.

    Parameters
    ----------
    bba_df : pd.DataFrame
        Output of :func:`parsers.parse_best_bid_ask`.
        Columns: ``best_bid``, ``bid_size``, ``best_ask``, ``ask_size``
        (prices in cents).

    Returns
    -------
    pd.DataFrame
        Columns: best_bid, best_ask, mid_price, spread,
        bid_vol_1, ask_vol_1, order_book_imbalance.
    """
    feats: dict[str, pd.Series] = {}

    # Prices:  cents → dollars
    feats["best_bid"] = bba_df["best_bid"] / 100.0
    feats["best_ask"] = bba_df["best_ask"] / 100.0
    feats["mid_price"] = (feats["best_bid"] + feats["best_ask"]) / 2.0
    feats["spread"] = feats["best_ask"] - feats["best_bid"]

    # Depth at level 1 (only level available from BEST_BID/ASK stream)
    feats["bid_vol_1"] = bba_df["bid_size"].fillna(0)
    feats["ask_vol_1"] = bba_df["ask_size"].fillna(0)

    # Order-book imbalance  (bid-heavy → > 0.5)
    total = feats["bid_vol_1"] + feats["ask_vol_1"]
    feats["order_book_imbalance"] = np.where(
        total > 0, feats["bid_vol_1"] / total, 0.5
    )

    return pd.DataFrame(feats, index=bba_df.index)


# ---------------------------------------------------------------------------
# Exchange (trade-stream) features
# ---------------------------------------------------------------------------

def compute_exchange_features(
    trade_df: pd.DataFrame,
    resample_freq: str = "1s",
) -> pd.DataFrame:
    """Aggregate trade-level data into per-interval features.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Output of :func:`parsers.parse_trades`.
        Columns: ``trade_price`` (cents), ``trade_qty``, ``is_buy``.
    resample_freq : str
        Pandas offset alias for the resample grid (default ``"1s"``).

    Returns
    -------
    pd.DataFrame
        Columns: last_trade_price (dollars), traded_volume, trade_intensity.
    """
    if trade_df.empty:
        return pd.DataFrame(
            columns=["last_trade_price", "traded_volume", "trade_intensity"],
            index=pd.DatetimeIndex([], name="time"),
        )

    trade_df = trade_df.copy()

    resampled = trade_df.resample(resample_freq)

    feats = pd.DataFrame(index=resampled.first().index)
    feats.index.name = "time"

    # Last executed trade price in each interval  (cents → dollars)
    feats["last_trade_price"] = resampled["trade_price"].last() / 100.0

    # Total volume traded per interval
    feats["traded_volume"] = resampled["trade_qty"].sum()

    # Number of trades per interval  (trade intensity / arrival rate)
    feats["trade_intensity"] = resampled["trade_qty"].count()

    return feats


# ---------------------------------------------------------------------------
# Execution-agent features
# ---------------------------------------------------------------------------

def compute_execution_features(
    exec_df: pd.DataFrame,
    total_inventory: int,
    mkt_open: pd.Timestamp,
    mkt_close: pd.Timestamp,
    resample_freq: str = "1s",
) -> pd.DataFrame:
    """Compute features tracking the execution agent's progress.

    If the execution agent didn't trade (empty log), returns a DataFrame
    full of default values (remaining_inventory = total_inventory, etc.).

    Parameters
    ----------
    exec_df : pd.DataFrame
        Output of :func:`parsers.load_execution_agent`.
    total_inventory : int
        Total shares the agent must liquidate.
    mkt_open, mkt_close : pd.Timestamp
        Market open / close times (for ``time_remaining``).
    resample_freq : str
        Resample frequency.

    Returns
    -------
    pd.DataFrame
        Columns: remaining_inventory, executed_volume,
        last_execution_price, time_remaining.
    """
    time_grid = pd.date_range(mkt_open, mkt_close, freq=resample_freq, name="time")
    feats = pd.DataFrame(index=time_grid)

    if exec_df.empty:
        feats["remaining_inventory"] = float(total_inventory)
        feats["executed_volume"] = 0.0
        feats["last_execution_price"] = np.nan
    else:
        # Resample cumulative quantities to the grid
        cum = exec_df["cumulative_qty"].resample(resample_freq).last()
        cum = cum.reindex(time_grid, method="ffill").fillna(0)
        feats["executed_volume"] = cum
        feats["remaining_inventory"] = total_inventory - cum

        # Convert exec prices: cents → dollars
        last_px = (exec_df["exec_price"] / 100.0).resample(resample_freq).last()
        last_px = last_px.reindex(time_grid, method="ffill")
        feats["last_execution_price"] = last_px

    # Time remaining: 1.0 at open → 0.0 at close
    total_seconds = (mkt_close - mkt_open).total_seconds()
    elapsed = (time_grid - mkt_open).total_seconds()
    feats["time_remaining"] = np.clip(1.0 - elapsed / total_seconds, 0.0, 1.0)

    return feats


# ---------------------------------------------------------------------------
# Derived / cross-source features
# ---------------------------------------------------------------------------

def compute_derived_features(
    merged_df: pd.DataFrame,
    fundamental: pd.Series | None = None,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Compute features that combine multiple data sources.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain at least ``mid_price``.  If ``last_trade_price``
        and ``traded_volume`` are present, VWAP is computed.
    fundamental : pd.Series or None
        Oracle fundamental value series (in cents) from
        :func:`parsers.load_fundamental`.
    volatility_window : int
        Rolling window size for return volatility.

    Returns
    -------
    pd.DataFrame
        Columns: volatility, vwap, benchmark_price.
    """
    feats: dict[str, pd.Series] = {}

    # --- Volatility: rolling std of log-returns of mid-price ---
    mid = merged_df["mid_price"]
    log_ret = np.log(mid / mid.shift(1))
    feats["volatility"] = log_ret.rolling(volatility_window, min_periods=1).std()

    # --- VWAP: cumulative volume-weighted average trade price ---
    if "last_trade_price" in merged_df.columns and "traded_volume" in merged_df.columns:
        px = merged_df["last_trade_price"].ffill()
        vol = merged_df["traded_volume"].fillna(0)
        cum_pv = (px * vol).cumsum()
        cum_v = vol.cumsum()
        feats["vwap"] = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
    else:
        feats["vwap"] = np.nan

    # --- Benchmark price: oracle fundamental value  (cents → dollars) ---
    if fundamental is not None and not fundamental.empty:
        fund = fundamental.reindex(merged_df.index, method="ffill")
        feats["benchmark_price"] = fund / 100.0
    else:
        feats["benchmark_price"] = np.nan

    return pd.DataFrame(feats, index=merged_df.index)
