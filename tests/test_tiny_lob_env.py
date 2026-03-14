import numpy as np

from execution_infra.tiny_lob_env import TinyLOBEnv


def test_tiny_lob_env_interface():
    env = TinyLOBEnv(n_steps=8, total_inventory=100, n_actions=5, seed=123)
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)
    for key in [
        "inventory_remaining",
        "remaining_inventory",
        "step",
        "exec_price",
        "mid_price",
        "executed_qty",
        "forced_qty",
        "step_is",
        "benchmark_price",
    ]:
        assert key in info

    next_obs, reward, terminated, truncated, next_info = env.step(1)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "executed_qty" in next_info
    assert "step_is" in next_info


def test_nonzero_action_trades_at_least_one_share_with_unit_inventory():
    env = TinyLOBEnv(n_steps=5, total_inventory=1, n_actions=15, seed=7)
    env.reset(seed=7)
    _, _, terminated, truncated, info = env.step(1)

    assert not truncated
    assert terminated
    assert info["executed_qty"] == 1
    assert info["inventory_remaining"] == 0


def test_wait_action_forces_completion_on_last_step():
    env = TinyLOBEnv(n_steps=1, total_inventory=10, n_actions=5, force_liquidation=True, seed=11)
    env.reset(seed=11)
    _, _, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert info["inventory_remaining"] == 0
    assert info["forced_qty"] == 10
    assert info["executed_qty"] == 10


def test_step_with_qty_executes_requested_amount():
    env = TinyLOBEnv(n_steps=5, total_inventory=20, n_actions=5, seed=17)
    env.reset(seed=17)
    _, _, terminated, truncated, info = env.step_with_qty(7)

    assert not terminated
    assert not truncated
    assert info["executed_qty"] == 7
    assert info["inventory_remaining"] == 13
