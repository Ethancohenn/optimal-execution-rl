"""
End-to-end feature extraction pipeline.

Reads ABIDES bz2 log files from a simulation run directory, computes
features, aligns them on either a uniform time grid (e.g. 1s) or native
event timestamps, and saves the result as a ``.npz`` file (or returns a
DataFrame).

CLI usage
---------
::

    python -m execution_infra.feature_extraction.pipeline \\
        --log-dir abides/log/my_sim_run \\
        --symbol IBM \\
        --freq 1s \\
        --output data/features.npz

Python usage
------------
::

    from execution_infra.feature_extraction import extract_features
    df, arrays = extract_features("abides/log/my_sim_run")
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from execution_infra.feature_extraction.parsers import (
    load_exchange,
    parse_best_bid_ask,
    parse_trades,
    load_execution_agent,
    load_fundamental,
)
from execution_infra.feature_extraction.features import (
    compute_orderbook_features,
    compute_exchange_features,
    compute_execution_features,
    compute_derived_features,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Parameters for the feature extraction pipeline.

    Attributes
    ----------
    symbol : str
        Ticker symbol used in the simulation (e.g. ``"IBM"``).
    resample_freq : str
        Pandas offset alias for the time grid (default ``"1s"``).
        Use ``"event"`` to keep native event timestamps.
    total_inventory : int
        Total shares the execution agent must liquidate.
    volatility_window : int
        Rolling window for return-volatility computation.
    """
    symbol: str = "IBM"
    resample_freq: str = "1s"
    total_inventory: int = 1_200_000
    volatility_window: int = 20


# ---------------------------------------------------------------------------
# Output column ordering (for documentation & consistency)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Orderbook (from BEST_BID / BEST_ASK events)
    "best_bid",
    "best_ask",
    "mid_price",
    "spread",
    "bid_vol_1",
    "ask_vol_1",
    "order_book_imbalance",
    # Exchange (from ORDER_EXECUTED events)
    "last_trade_price",
    "traded_volume",
    "trade_intensity",
    # Execution agent
    "remaining_inventory",
    "executed_volume",
    "last_execution_price",
    "time_remaining",
    # Derived
    "volatility",
    "vwap",
    "benchmark_price",
]

EVENT_FREQ_TOKENS = {"event", "none", "raw", "native", "ns", "nanosecond", "nanoseconds"}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_features(
    log_dir: str | Path,
    config: FeatureConfig | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run the full feature-extraction pipeline.

    Parameters
    ----------
    log_dir : str or Path
        Path to the ABIDES log directory (e.g. ``"abides/log/my_sim_run"``).
    config : FeatureConfig or None
        Pipeline configuration.  Uses defaults if ``None``.
    output_path : str, Path, or None
        If provided, saves features to a ``.npz`` file at this path.

    Returns
    -------
    df : pd.DataFrame
        The final feature DataFrame (DatetimeIndex, 17 columns).
    arrays : dict[str, np.ndarray]
        ``{"features": ..., "feature_names": ..., "timestamps": ..., "timestamps_ns": ...}``
    """
    cfg = config or FeatureConfig()
    log_dir = Path(log_dir)

    # ── 1. Locate files ──────────────────────────────────────────
    # ABIDES log naming varies across configs/versions.
    ex_candidates = [
        log_dir / "EXCHANGE_AGENT.bz2",
        log_dir / "ExchangeAgent.bz2",
        log_dir / "ExchangeAgent0.bz2",
    ]
    ex_path = next((p for p in ex_candidates if p.exists()), ex_candidates[0])
    fund_path = log_dir / f"fundamental_{cfg.symbol}.bz2"

    # Execution agent name may vary; find the first match
    exec_candidates = (
        list(log_dir.glob("*EXECUTION*.bz2"))
        + list(log_dir.glob("*execution*.bz2"))
    )
    exec_path = exec_candidates[0] if exec_candidates else None

    # ── 2. Parse the exchange stream (primary data source) ────────
    print(f"[1/5] Loading exchange stream from {ex_path.name} ...")
    exchange_raw = load_exchange(ex_path)

    print("[2/5] Parsing best bid/ask & trades ...")
    bba_df = parse_best_bid_ask(exchange_raw)
    trade_df = parse_trades(exchange_raw)

    # ── 3. Parse secondary logs ──────────────────────────────────
    print("[3/5] Loading fundamental values & execution agent ...")
    fund_raw = (
        load_fundamental(fund_path)
        if fund_path.exists()
        else pd.Series(dtype=float, name="fundamental_value")
    )
    if isinstance(fund_raw.index, pd.DatetimeIndex) and fund_raw.index.has_duplicates:
        fund_raw = fund_raw.groupby(level=0).last().sort_index()

    if exec_path and exec_path.exists():
        exec_raw = load_execution_agent(exec_path)
    else:
        exec_raw = pd.DataFrame(
            columns=["exec_price", "exec_qty", "cumulative_qty"],
            index=pd.DatetimeIndex([], name="time"),
        )

    # ── 4. Compute feature groups ─────────────────────────────────
    print("[4/5] Computing features ...")

    # Orderbook features → resample to uniform grid
    ob_feats = compute_orderbook_features(bba_df)
    freq_mode = str(cfg.resample_freq).lower()
    use_event_grid = freq_mode in EVENT_FREQ_TOKENS

    if use_event_grid:
        # Some ABIDES streams can emit multiple rows with identical timestamps.
        # Collapse to one row per timestamp so reindex(..., method="ffill") is valid.
        ob_event = ob_feats.groupby(level=0).last().sort_index()

        # Event grid: sorted union of all available event timestamps.
        idx_parts: list[pd.DatetimeIndex] = []
        for idx in (ob_event.index, trade_df.index, exec_raw.index, fund_raw.index):
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                idx_parts.append(idx)
        if not idx_parts:
            raise ValueError("No timestamps found in logs; cannot build event-level feature grid.")

        event_index = idx_parts[0]
        for idx in idx_parts[1:]:
            event_index = event_index.union(idx)
        event_index = event_index.sort_values().unique()
        ob_grid = ob_event.reindex(event_index, method="ffill")
    else:
        # Uniform time grid (default 1s).
        ob_grid = ob_feats.resample(cfg.resample_freq).last().ffill()

    # Exchange features (event-level or internally resampled)
    ex_feats = compute_exchange_features(
        trade_df,
        resample_freq=None if use_event_grid else cfg.resample_freq,
    )

    # Market open / close from the data
    mkt_open = bba_df.index.min()
    mkt_close = bba_df.index.max()

    # Execution features
    exec_feats = compute_execution_features(
        exec_raw,
        total_inventory=cfg.total_inventory,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        resample_freq=cfg.resample_freq,
        time_index=ob_grid.index if use_event_grid else None,
    )

    # ── 5. Merge on time grid ─────────────────────────────────────
    merged = ob_grid.copy()

    # Join exchange features
    if not ex_feats.empty:
        merged = merged.join(ex_feats, how="left")
    else:
        merged["last_trade_price"] = np.nan
        merged["traded_volume"] = 0.0
        merged["trade_intensity"] = 0.0

    # Join execution features
    merged = merged.join(exec_feats, how="left", rsuffix="_exec")

    # Forward-fill prices, zero-fill volumes
    for col in ["last_trade_price", "last_execution_price"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    for col in ["traded_volume", "trade_intensity"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # ── 6. Derived features ───────────────────────────────────────
    derived = compute_derived_features(
        merged,
        fundamental=fund_raw if not fund_raw.empty else None,
        volatility_window=cfg.volatility_window,
    )
    merged = merged.join(derived)

    # ── 7. Final cleanup ──────────────────────────────────────────
    available = [c for c in FEATURE_COLUMNS if c in merged.columns]
    df = merged[available].copy()
    df = df.dropna(subset=["mid_price"])

    print(f"[5/5] Done.  Shape: {df.shape}  |  Columns: {list(df.columns)}")

    # ── 8. Build numpy arrays ─────────────────────────────────────
    feature_names = np.array(df.columns.tolist())
    # Preserve native precision (including ns) in both string and integer forms.
    timestamps = df.index.astype("datetime64[ns]").astype(str).to_numpy()
    timestamps_ns = df.index.astype("datetime64[ns]").astype("int64").to_numpy()
    features = df.values.astype(np.float32)

    arrays = {
        "features": features,
        "feature_names": feature_names,
        "timestamps": timestamps,
        "timestamps_ns": timestamps_ns,
    }

    # ── 9. Optionally save ────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), **arrays)
        print(f"Saved to {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    return df, arrays


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract RL-ready features from ABIDES simulation logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m execution_infra.feature_extraction.pipeline \\
      --log-dir abides/log/my_sim_run \\
      --symbol IBM --freq 1s \\
      --output data/features.npz
        """,
    )
    parser.add_argument("--log-dir", required=True, help="Path to ABIDES log directory")
    parser.add_argument("--symbol", default="IBM", help="Ticker symbol (default: IBM)")
    parser.add_argument(
        "--freq",
        default="1s",
        help="Resample frequency (default: 1s). Use 'event' to keep native timestamps.",
    )
    parser.add_argument("--total-inventory", type=int, default=1_200_000,
                        help="Execution agent total inventory")
    parser.add_argument("--output", default=None, help="Output path for .npz file")

    args = parser.parse_args()

    cfg = FeatureConfig(
        symbol=args.symbol,
        resample_freq=args.freq,
        total_inventory=args.total_inventory,
    )

    df, _ = extract_features(args.log_dir, config=cfg, output_path=args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("Feature Summary")
    print("=" * 60)
    print(f"Shape:      {df.shape}")
    print(f"Time range: {df.index[0]} → {df.index[-1]}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\n{df.describe().to_string()}")


if __name__ == "__main__":
    main()
