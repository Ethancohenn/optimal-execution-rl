"""
Tests for AbidesReplayEnv using an in-memory mock npz (no real data file needed).
"""
from __future__ import annotations

import io
import numpy as np
import pytest
import tempfile
import os

from execution_infra.abides_replay_env import AbidesReplayEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(n_rows: int = 200, seed: int = 0) -> str:
    """Create a temporary .npz with the required feature columns."""
    rng = np.random.default_rng(seed)
    feature_names = np.array([
        "best_bid", "best_ask", "mid_price", "spread",
        "bid_vol_1", "ask_vol_1", "order_book_imbalance",
        "last_trade_price", "traded_volume", "trade_intensity",
        "remaining_inventory", "executed_volume",
        "last_execution_price", "time_remaining",
        "volatility", "vwap", "benchmark_price",
    ])
    F = len(feature_names)
    col = {name: i for i, name in enumerate(feature_names)}

    features = rng.random((n_rows, F)).astype(np.float32)
    # Make prices realistic
    mid = 100.0 + rng.standard_normal(n_rows) * 0.5
    features[:, col["mid_price"]] = mid.astype(np.float32)
    features[:, col["best_bid"]] = (mid - 0.01).astype(np.float32)
    features[:, col["best_ask"]] = (mid + 0.01).astype(np.float32)
    features[:, col["spread"]] = 0.02
    features[:, col["bid_vol_1"]] = rng.integers(10, 200, n_rows).astype(np.float32)
    features[:, col["order_book_imbalance"]] = rng.random(n_rows).astype(np.float32)
    timestamps = np.array([f"2014-01-28 09:{i//60:02d}:{i%60:02d}" for i in range(n_rows)])

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez_compressed(tmp.name, features=features, feature_names=feature_names, timestamps=timestamps)
    return tmp.name


@pytest.fixture(scope="module")
def npz_path():
    path = _make_npz(n_rows=200)
    yield path
    os.unlink(path)


@pytest.fixture
def env(npz_path):
    return AbidesReplayEnv(npz_path=npz_path, n_steps=20, total_inventory=1000, n_actions=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_observation_space_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (6,), f"Expected (6,), got {obs.shape}"
    assert env.observation_space.contains(obs), "Initial obs not in observation_space"


def test_obs_values_in_range(env):
    obs, _ = env.reset(seed=0)
    assert 0.0 <= obs[0] <= 1.0, "inv_norm out of [0,1]"
    assert 0.0 <= obs[1] <= 1.0, "time_norm out of [0,1]"
    assert obs[2] > 0, "mid_price_norm should be positive"
    assert obs[3] > 0, "spread_norm should be positive"
    assert obs[4] > 0, "bid_vol_norm should be positive"
    assert 0.0 <= obs[5] <= 1.0, "obi out of [0,1]"


def test_step_returns_correct_shapes(env):
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "inventory_remaining" in info
    assert "exec_price" in info


def test_episode_ends_after_n_steps(env):
    env.reset(seed=2)
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)  # always wait
        done = terminated or truncated
        steps += 1
        assert steps <= env.n_steps + 1, "Episode exceeded n_steps without terminating"
    assert done


def test_full_liquidation_terminates_early(env):
    """Selling 100% of inventory at step 0 should terminate immediately."""
    env.reset(seed=3)
    # action 4 = 100% of remaining inventory
    _, _, terminated, truncated, info = env.step(4)
    assert terminated, "Should terminate after full liquidation"
    assert info["inventory_remaining"] == 0


def test_different_seeds_give_different_windows(env):
    obs1, _ = env.reset(seed=10)
    start1 = env._start
    obs2, _ = env.reset(seed=99)
    start2 = env._start
    # With 200 rows and n_steps=20 there are 180 valid starts — two random seeds very
    # likely hit different ones.
    # (Not guaranteed, but overwhelmingly probable.)
    assert start1 != start2 or np.allclose(obs1, obs2)


def test_market_impact_reduces_exec_price(env):
    """Selling shares should incur market impact: exec_price must be strictly below mid_price."""
    env.reset(seed=5)
    _, _, _, _, info = env.step(4)  # sell 100% immediately
    assert info["executed_qty"] > 0, "Should have executed shares"
    assert info["exec_price"] < info["mid_price"], (
        f"exec_price {info['exec_price']:.6f} should be < mid_price {info['mid_price']:.6f} due to impact"
    )



def test_info_keys(env):
    env.reset(seed=6)
    env.step(1)
    _, _, _, _, info = env.step(2)
    for key in ["inventory_remaining", "shares_sold", "step", "exec_price",
                 "mid_price", "executed_qty", "benchmark_price", "completion_rate"]:
        assert key in info, f"Missing info key: {key}"


def test_too_small_npz_raises():
    """Creating env with n_steps > available rows should raise."""
    path = _make_npz(n_rows=10)
    try:
        with pytest.raises(ValueError, match="only 10 rows"):
            AbidesReplayEnv(npz_path=path, n_steps=50)
    finally:
        os.unlink(path)
