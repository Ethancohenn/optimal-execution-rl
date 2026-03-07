import numpy as np
import torch

from src.agents.dqn import DQNAgent


def test_dqn_action_in_range():
    agent = DQNAgent(
        state_dim=6,
        action_dim=5,
        hidden_dims=[32, 32],
        warmup_steps=1,
        batch_size=1,
    )
    action = agent.select_action(np.zeros(6, dtype=np.float32), greedy=True)
    assert 0 <= action < 5


def test_double_dqn_targets_use_policy_argmax():
    agent = DQNAgent(
        state_dim=2,
        action_dim=3,
        hidden_dims=[8, 8],
        double_dqn=True,
        gamma=1.0,
        warmup_steps=1,
        batch_size=1,
    )

    # Policy picks action 0 at next state, target would prefer action 1 if maxed directly.
    agent.policy_net = torch.nn.Linear(2, 3, bias=False)
    agent.target_net = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        agent.policy_net.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
        agent.target_net.weight.copy_(torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]))

    next_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    rewards = torch.tensor([0.0], dtype=torch.float32)
    dones = torch.tensor([0.0], dtype=torch.float32)
    targets = agent._compute_targets(rewards=rewards, dones=dones, next_states=next_states)

    # Double DQN uses target Q at policy argmax action=0 -> value 0.0.
    assert torch.allclose(targets, torch.tensor([0.0]))


def test_smoothness_loss_positive_for_large_action_jump():
    agent = DQNAgent(
        state_dim=2,
        action_dim=5,
        hidden_dims=[8, 8],
        smoothness_coef=1.0,
        warmup_steps=1,
        batch_size=1,
    )
    q_values = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]], dtype=torch.float32)
    prev_actions = torch.tensor([0.0], dtype=torch.float32)
    loss = agent._smoothness_loss(q_values, prev_actions)
    assert float(loss.item()) > 0.0
