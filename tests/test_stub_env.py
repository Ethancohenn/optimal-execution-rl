import numpy as np

from src.envs.stub_env import StubExecutionEnv

def test_stub_env_interface():
    env = StubExecutionEnv(T=5, Q0=100.0, seed=123)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    for k in ["executed_qty", "exec_price", "mid_price", "spread", "remaining_inventory", "t"]:
        assert k in info

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    for k in ["executed_qty", "exec_price", "mid_price", "spread", "remaining_inventory", "t"]:
        assert k in info2
