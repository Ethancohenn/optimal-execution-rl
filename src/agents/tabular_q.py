from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscretizerSpec:
    bins: tuple[int, ...] = (12, 12, 2, 2, 2, 2)
    lows: tuple[float, ...] = (0.0, 0.0, 0.0, -1.0, -1.0, 0.0)
    highs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "DiscretizerSpec":
        default = cls()
        raw = raw or {}
        return cls(
            bins=tuple(int(x) for x in raw.get("bins", default.bins)),
            lows=tuple(float(x) for x in raw.get("lows", default.lows)),
            highs=tuple(float(x) for x in raw.get("highs", default.highs)),
        )


class ObservationDiscretizer:
    def __init__(self, spec: DiscretizerSpec | None = None):
        self.spec = spec or DiscretizerSpec()
        self.bins = tuple(int(x) for x in self.spec.bins)
        self.lows = np.asarray(self.spec.lows, dtype=np.float32)
        self.highs = np.asarray(self.spec.highs, dtype=np.float32)
        if not (len(self.bins) == len(self.lows) == len(self.highs)):
            raise ValueError("DiscretizerSpec dimensions must match.")

    def transform(self, obs: np.ndarray) -> tuple[int, ...]:
        obs = np.asarray(obs, dtype=np.float32)
        clipped = np.clip(obs, self.lows, self.highs)
        widths = np.maximum(self.highs - self.lows, 1e-8)
        ratios = (clipped - self.lows) / widths

        indices = []
        for ratio, n_bins in zip(ratios, self.bins):
            idx = int(np.floor(ratio * n_bins))
            indices.append(min(n_bins - 1, max(0, idx)))
        return tuple(indices)

    def metadata(self) -> dict:
        return {
            "bins": list(self.bins),
            "lows": [float(x) for x in self.lows],
            "highs": [float(x) for x in self.highs],
        }


def sample_greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    q_values = np.asarray(q_values, dtype=np.float32)
    max_q = float(np.max(q_values))
    max_actions = np.flatnonzero(np.isclose(q_values, max_q))
    return int(rng.choice(max_actions))


class TabularQAgent:
    def __init__(
        self,
        state_bins: tuple[int, ...],
        action_dim: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.state_bins = tuple(int(x) for x in state_bins)
        self.action_dim = int(action_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.q_table = np.zeros((*self.state_bins, self.action_dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.train_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: tuple[int, ...], greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon():
            return int(self.rng.integers(0, self.action_dim))
        return sample_greedy_action(self.q_table[state], self.rng)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> float:
        bootstrap = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = float(reward) + self.gamma * bootstrap
        td_error = td_target - float(self.q_table[state][action])
        self.q_table[state][action] += self.alpha * td_error
        self.train_steps += 1
        return float(td_error)

    def save(self, path: str) -> None:
        np.save(path, self.q_table)
