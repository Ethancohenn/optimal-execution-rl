from __future__ import annotations

import numpy as np

from execution_infra import EnvConfig, ExecutionEnv
from src.agents.tabular_q import DiscretizerSpec, ObservationDiscretizer, TabularQAgent


def _deterministic_config(**overrides) -> EnvConfig:
    base = dict(
        total_inventory=100,
        n_steps=4,
        n_actions=5,
        initial_price=100.0,
        tick_size=0.01,
        spread_mean=0.02,
        spread_std=0.0,
        lob_depth_mean=40,
        lob_depth_std=0,
        n_lob_levels=4,
        depth_decay=0.0,
        obs_depth_levels=3,
        market_lot_size=10,
        market_order_intensity=0.0,
        limit_order_intensity=0.0,
        cancellation_rate=0.0,
        quote_improve_prob=0.0,
        volatility=0.0,
        fundamental_mean_reversion=0.0,
        order_flow_memory=0.0,
        agent_permanent_impact=0.0,
        force_liquidation=True,
        seed=123,
    )
    base.update(overrides)
    return EnvConfig(**base)


def test_execution_env_observation_shape_and_bounds():
    env = ExecutionEnv(_deterministic_config())
    obs, _ = env.reset(seed=11)

    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)


def test_step_with_qty_executes_exact_quantity():
    env = ExecutionEnv(_deterministic_config(force_liquidation=False))
    env.reset(seed=7)

    _, _, _, _, info = env.step_with_qty(37)

    assert info["executed_qty"] == 37
    assert info["inventory_remaining"] == 63


def test_sell_action_changes_future_market_state():
    cfg = _deterministic_config(force_liquidation=False)

    wait_env = ExecutionEnv(cfg)
    wait_env.reset(seed=5)
    _, _, _, _, wait_info = wait_env.step(0)

    sell_env = ExecutionEnv(cfg)
    sell_env.reset(seed=5)
    _, _, _, _, sell_info = sell_env.step_with_qty(60)

    assert sell_info["next_mid_price"] < wait_info["next_mid_price"]
    assert sell_info["spread"] >= wait_info["spread"]


def test_force_liquidation_finishes_horizon():
    env = ExecutionEnv(_deterministic_config(n_steps=2, total_inventory=80, force_liquidation=True))
    env.reset(seed=3)

    env.step(0)
    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert info["forced_qty"] > 0
    assert info["inventory_remaining"] == 0


def test_tabular_q_agent_updates_q_table():
    discretizer = ObservationDiscretizer()
    agent = TabularQAgent(state_bins=discretizer.bins, action_dim=5, seed=0)

    state = discretizer.transform(np.array([1.0, 1.0, 0.2, 0.0, 0.0, 0.5], dtype=np.float32))
    next_state = discretizer.transform(np.array([0.8, 0.75, 0.3, -0.2, 0.1, 0.4], dtype=np.float32))

    before = float(agent.q_table[state][2])
    agent.update(state, 2, reward=1.0, next_state=next_state, done=False)
    after = float(agent.q_table[state][2])

    assert after != before
    assert 0 <= agent.select_action(state, greedy=True) < 5


def test_tabular_q_agent_breaks_greedy_ties_without_action_zero_bias():
    discretizer = ObservationDiscretizer()
    agent = TabularQAgent(state_bins=discretizer.bins, action_dim=5, seed=0)
    state = discretizer.transform(np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.5], dtype=np.float32))

    actions = {agent.select_action(state, greedy=True) for _ in range(32)}

    assert actions != {0}
    assert len(actions) > 1


def test_discretizer_spec_from_dict_round_trips_metadata():
    spec = DiscretizerSpec.from_dict(
        {
            "bins": [9, 7, 2, 3, 4, 5],
            "lows": [0.0, 0.0, 0.1, -1.0, -0.5, 0.2],
            "highs": [1.0, 1.0, 0.9, 1.0, 0.5, 0.8],
        }
    )

    assert spec.bins == (9, 7, 2, 3, 4, 5)
    assert spec.lows == (0.0, 0.0, 0.1, -1.0, -0.5, 0.2)
    assert spec.highs == (1.0, 1.0, 0.9, 1.0, 0.5, 0.8)
