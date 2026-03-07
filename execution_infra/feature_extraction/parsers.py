"""
Low-level readers for ABIDES bz2-compressed log files.

Each function reads one log file and returns a cleaned pandas DataFrame
ready for feature engineering.

ABIDES logs are bz2-compressed pickled DataFrames written by the Kernel
at simulation end.  The four log types handled here:

    EXCHANGE_AGENT.bz2            – exchange event stream (richest source)
    fundamental_{SYMBOL}.bz2      – oracle fundamental value series
    {EXECUTION_AGENT_NAME}.bz2    – execution agent event log
    ORDERBOOK_{SYMBOL}_FULL.bz2   – order-book snapshots (legacy, optional)

.. note::

    The ``ORDERBOOK_*_FULL.bz2`` file is pickled with an old pandas version
    that used ``SparseDataFrame``.  On modern pandas (≥1.0) this file is
    extremely slow or impossible to unpickle.  Therefore, **order-book
    features are extracted from the exchange stream** (``BEST_BID``,
    ``BEST_ASK`` events) which is faster and always loadable.
"""

from __future__ import annotations

import bz2
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exchange agent log  →  best-bid/ask + trades
# ---------------------------------------------------------------------------

def load_exchange(path: str | Path) -> pd.DataFrame:
    """Load the exchange agent log.

    This is the **primary data source** for the pipeline.  It contains
    all order-book events (``BEST_BID``, ``BEST_ASK``), executed trades
    (``ORDER_EXECUTED``, ``LAST_TRADE``), and other activity.

    Parameters
    ----------
    path : str or Path
        Path to ``EXCHANGE_AGENT.bz2``.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with ``EventTime`` index, ``EventType``, ``Event``.
    """
    with bz2.open(str(path), "rb") as f:
        raw = pickle.load(f)
    return raw


def parse_best_bid_ask(exchange_df: pd.DataFrame) -> pd.DataFrame:
    """Extract best-bid and best-ask from the exchange stream.

    ABIDES logs ``BEST_BID`` / ``BEST_ASK`` events as strings with
    format ``"SYMBOL,PRICE,SIZE"`` (price in cents, size in shares).

    Parameters
    ----------
    exchange_df : pd.DataFrame
        Raw exchange log from :func:`load_exchange`.

    Returns
    -------
    pd.DataFrame
        Indexed by time with columns:
        ``best_bid``, ``bid_size``, ``best_ask``, ``ask_size``
        (prices in **cents**).
    """
    bid_rows = exchange_df[exchange_df["EventType"] == "BEST_BID"].copy()
    ask_rows = exchange_df[exchange_df["EventType"] == "BEST_ASK"].copy()

    def _parse_bba(df: pd.DataFrame, price_col: str, size_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[price_col, size_col],
                                index=pd.DatetimeIndex([], name="time"))
        parts = df["Event"].str.split(",", expand=True)
        result = pd.DataFrame({
            price_col: pd.to_numeric(parts[1], errors="coerce"),
            size_col: pd.to_numeric(parts[2], errors="coerce"),
        }, index=df.index)
        result.index.name = "time"
        return result

    bids = _parse_bba(bid_rows, "best_bid", "bid_size")
    asks = _parse_bba(ask_rows, "best_ask", "ask_size")

    # Merge on time, forward-filling so every row has both sides
    merged = bids.join(asks, how="outer").sort_index().ffill()
    return merged


def parse_trades(exchange_df: pd.DataFrame) -> pd.DataFrame:
    """Extract executed trades from the exchange stream.

    Uses ``ORDER_EXECUTED`` events which contain dicts with
    ``fill_price`` (cents), ``quantity``, ``is_buy_order``.

    Parameters
    ----------
    exchange_df : pd.DataFrame
        Raw exchange log from :func:`load_exchange`.

    Returns
    -------
    pd.DataFrame
        Indexed by time with columns:
        ``trade_price`` (cents), ``trade_qty``, ``is_buy``.
    """
    executed = exchange_df[exchange_df["EventType"] == "ORDER_EXECUTED"]
    if executed.empty:
        return pd.DataFrame(
            columns=["trade_price", "trade_qty", "is_buy"],
            index=pd.DatetimeIndex([], name="time"),
        )

    records: list[dict] = []
    for ts, row in executed.iterrows():
        evt = row["Event"]
        if isinstance(evt, dict):
            records.append({
                "time": ts,
                "trade_price": evt.get("fill_price", evt.get("limit_price", np.nan)),
                "trade_qty": evt.get("quantity", 0),
                "is_buy": bool(evt.get("is_buy_order", False)),
            })

    if not records:
        return pd.DataFrame(
            columns=["trade_price", "trade_qty", "is_buy"],
            index=pd.DatetimeIndex([], name="time"),
        )

    df = pd.DataFrame(records).set_index("time")
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Execution agent log
# ---------------------------------------------------------------------------

def load_execution_agent(path: str | Path) -> pd.DataFrame:
    """Load the execution agent's log and extract fill events.

    Parameters
    ----------
    path : str or Path
        Path to e.g. ``POV_EXECUTION_AGENT.bz2``.

    Returns
    -------
    pd.DataFrame
        Indexed by time with columns:
        ``exec_price`` (cents), ``exec_qty``, ``cumulative_qty``.
    """
    with bz2.open(str(path), "rb") as f:
        raw = pickle.load(f)

    executed = raw[raw["EventType"] == "ORDER_EXECUTED"].copy()
    if executed.empty:
        return pd.DataFrame(
            columns=["exec_price", "exec_qty", "cumulative_qty"],
            index=pd.DatetimeIndex([], name="time"),
        )

    records: list[dict] = []
    cum = 0
    for ts, row in executed.iterrows():
        evt = row["Event"]
        if isinstance(evt, dict):
            qty = evt.get("quantity", 0)
            cum += qty
            records.append({
                "time": ts,
                "exec_price": evt.get("fill_price", evt.get("limit_price", np.nan)),
                "exec_qty": qty,
                "cumulative_qty": cum,
            })

    if not records:
        return pd.DataFrame(
            columns=["exec_price", "exec_qty", "cumulative_qty"],
            index=pd.DatetimeIndex([], name="time"),
        )

    df = pd.DataFrame(records).set_index("time")
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Fundamental value log
# ---------------------------------------------------------------------------

def load_fundamental(path: str | Path) -> pd.Series:
    """Load oracle fundamental value series.

    Parameters
    ----------
    path : str or Path
        Path to ``fundamental_{SYMBOL}.bz2``.

    Returns
    -------
    pd.Series
        Named ``fundamental_value`` (in cents), indexed by time.
    """
    with bz2.open(str(path), "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, pd.DataFrame):
        if "FundamentalTime" in raw.columns:
            raw = raw.set_index("FundamentalTime")
        # Take the value column (usually 'FundamentalValue')
        val_cols = [c for c in raw.columns if "value" in c.lower() or "fundamental" in c.lower()]
        col = val_cols[0] if val_cols else raw.columns[0]
        series = raw[col].rename("fundamental_value")
    else:
        series = raw.rename("fundamental_value")

    series.index.name = "time"
    series.sort_index(inplace=True)
    return series
