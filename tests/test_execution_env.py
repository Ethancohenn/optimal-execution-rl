from execution_infra import EnvConfig, ExecutionEnv


def test_nonzero_action_trades_at_least_one_share_with_unit_inventory():
    cfg = EnvConfig(total_inventory=1, n_steps=5, n_actions=15, seed=123)
    env = ExecutionEnv(config=cfg)
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(1)

    assert not truncated
    assert terminated
    assert info["shares_sold"] == 1
    assert info["inventory_remaining"] == 0


def test_nonzero_action_liquidates_all_on_final_step():
    cfg = EnvConfig(total_inventory=2, n_steps=1, n_actions=15, seed=123)
    env = ExecutionEnv(config=cfg)
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(1)

    assert not truncated
    assert terminated
    assert info["shares_sold"] == 2
    assert info["inventory_remaining"] == 0
