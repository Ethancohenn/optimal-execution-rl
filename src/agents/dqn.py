from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
from typing import Deque, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        if len(hidden_dims) < 2 or len(hidden_dims) > 3:
            raise ValueError("hidden_dims must have length 2 or 3")

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    prev_action: int


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_action: int,
    ) -> None:
        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                prev_action=int(prev_action),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buffer, batch_size)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        warmup_steps: int = 1_000,
        target_update_interval: int = 50,
        tau: float = 0.05,
        smoothness_coef: float = 0.0,
        double_dqn: bool = True,
        td_loss: str = "huber",
        max_grad_norm: float | None = 10.0,
        device: str | None = None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = hidden_dims or [128, 128]
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        self.tau = float(tau)
        self.smoothness_coef = float(smoothness_coef)
        self.double_dqn = bool(double_dqn)
        self.td_loss = str(td_loss).lower()
        if self.td_loss not in {"mse", "huber"}:
            raise ValueError("td_loss must be 'mse' or 'huber'")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=replay_capacity)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)

        self.train_steps = 0
        self.num_actions = torch.arange(self.action_dim, dtype=torch.float32, device=self.device)

    def epsilon(self) -> float:
        progress = min(1.0, self.train_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_action: int,
    ) -> None:
        self.replay.add(state, action, reward, next_state, done, prev_action)

    def _smoothness_loss(self, q_values: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        if self.smoothness_coef <= 0.0:
            return torch.tensor(0.0, device=self.device)
        policy_probs = F.softmax(q_values, dim=1)
        expected_action = torch.sum(policy_probs * self.num_actions.unsqueeze(0), dim=1)
        denom = max(1.0, float(self.action_dim - 1))
        return torch.mean(((expected_action - prev_actions) / denom) ** 2)

    def _compute_targets(
        self, rewards: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.double_dqn:
                next_policy_q = self.policy_net(next_states)
                next_actions = torch.argmax(next_policy_q, dim=1, keepdim=True)
                next_target_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_target_q = self.target_net(next_states).max(dim=1).values
            return rewards + (1.0 - dones) * self.gamma * next_target_q

    def train_step(self) -> dict | None:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None

        batch = self.replay.sample(self.batch_size)

        states = torch.as_tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)
        prev_actions = torch.as_tensor(
            [float(t.prev_action) for t in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.policy_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_targets = self._compute_targets(rewards, dones, next_states)

        if self.td_loss == "huber":
            td_loss = F.smooth_l1_loss(q_sa, td_targets)
        else:
            td_loss = F.mse_loss(q_sa, td_targets)
        smooth_loss = self._smoothness_loss(q_values, prev_actions)
        loss = td_loss + self.smoothness_coef * smooth_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.soft_update_target()

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "smoothness_loss": float(smooth_loss.item()),
            "epsilon": float(self.epsilon()),
        }

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * policy_param.data)
