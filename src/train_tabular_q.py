from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra.light_lob_env import LightLOBEnv
from execution_infra.tiny_lob_env import TinyLOBEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.common.logger import RunLogger


@dataclass(frozen=True)
class DiscretizerSpec:
    bins: np.ndarray
    lows: np.ndarray
    highs: np.ndarray


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool, prefix: str) -> str:
    if run_name is None:
        run_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            try:
                shutil.rmtree(run_dir)
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot overwrite run directory '{run_dir}'. "
                    "Close files under this folder (editor/Explorer) or use a different run name."
                ) from exc
        else:
            raise ValueError(f"Run directory already exists: {run_dir}. Use --overwrite or a new run name.")
    return run_dir


def epsilon_for_episode(ep: int, start: float, end: float, decay_episodes: int) -> float:
    if decay_episodes <= 0:
        return float(end)
    frac = min(1.0, ep / float(decay_episodes))
    return float(start + frac * (end - start))


def make_discretizer(args: argparse.Namespace) -> DiscretizerSpec:
    bins = np.array(
        [
            args.inv_bins,
            args.time_bins,
            args.mid_bins,
            args.spread_bins,
            args.vol_bins,
            args.obi_bins,
        ],
        dtype=np.int32,
    )
    lows = np.array([0.0, 0.0, args.mid_low, 0.0, 0.0, 0.0], dtype=np.float32)
    highs = np.array([1.0, 1.0, args.mid_high, args.spread_high, args.vol_high, 1.0], dtype=np.float32)
    if np.any(bins <= 0):
        raise ValueError("All discretization bins must be >= 1.")
    if np.any(highs <= lows):
        raise ValueError("Discretization highs must be > lows for all dimensions.")
    return DiscretizerSpec(bins=bins, lows=lows, highs=highs)


def encode_obs(obs: np.ndarray, spec: DiscretizerSpec) -> tuple[int, int, int, int, int, int]:
    clipped = np.clip(obs.astype(np.float32), spec.lows, spec.highs)
    frac = (clipped - spec.lows) / (spec.highs - spec.lows)
    idx = np.floor(frac * spec.bins).astype(np.int32)
    idx = np.clip(idx, 0, spec.bins - 1)
    return tuple(int(v) for v in idx)


def select_action(
    q_table: np.ndarray,
    state: tuple[int, int, int, int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    q_values = q_table[state]
    max_q = float(np.max(q_values))
    max_actions = np.flatnonzero(np.isclose(q_values, max_q))
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[-1]))
    return int(rng.choice(max_actions))


def greedy_action(
    q_table: np.ndarray,
    state: tuple[int, int, int, int, int, int],
    rng: np.random.Generator,
) -> int:
    q_values = q_table[state]
    max_q = float(np.max(q_values))
    max_actions = np.flatnonzero(np.isclose(q_values, max_q))
    return int(rng.choice(max_actions))


def save_q_table(
    out_path: str,
    q_table: np.ndarray,
    spec: DiscretizerSpec,
    norm_stats: dict[str, float],
    n_actions: int,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        q_table=q_table.astype(np.float32),
        bins=spec.bins.astype(np.int32),
        lows=spec.lows.astype(np.float32),
        highs=spec.highs.astype(np.float32),
        mean_spread=np.float32(norm_stats["mean_spread"]),
        mean_bid_vol=np.float32(norm_stats["mean_bid_vol"]),
        n_actions=np.int32(n_actions),
    )


def load_q_table(path: str) -> tuple[np.ndarray, DiscretizerSpec, dict[str, float], int]:
    data = np.load(path)
    q_table = data["q_table"].astype(np.float32)
    spec = DiscretizerSpec(
        bins=data["bins"].astype(np.int32),
        lows=data["lows"].astype(np.float32),
        highs=data["highs"].astype(np.float32),
    )
    norm_stats = {
        "mean_spread": float(data["mean_spread"]),
        "mean_bid_vol": float(data["mean_bid_vol"]),
    }
    n_actions = int(data["n_actions"])
    return q_table, spec, norm_stats, n_actions


def _step_is(info: dict) -> float:
    executed_qty = float(info.get("executed_qty", 0.0))
    exec_price = float(info.get("exec_price", float("nan")))
    mid_price = float(info.get("mid_price", float("nan")))
    if executed_qty > 0 and not (np.isnan(exec_price) or np.isnan(mid_price)):
        return float(executed_qty * (mid_price - exec_price))
    return 0.0


def _resolve_env_name(args: argparse.Namespace) -> str:
    return "abides" if args.use_abides else str(args.env)


def _build_env(
    args: argparse.Namespace,
    split: str,
    norm_stats: dict[str, float] | None = None,
) -> tuple[object, str, dict[str, float]]:
    env_name = _resolve_env_name(args)
    if env_name == "abides":
        env = AbidesReplayEnv(
            npz_path=args.npz_path,
            n_steps=args.n_steps,
            total_inventory=args.total_inventory,
            n_actions=args.n_actions,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            split=split,
            norm_stats=norm_stats,
        )
        if norm_stats is None:
            stats = {"mean_spread": env._mean_spread, "mean_bid_vol": env._mean_bid_vol}
        else:
            stats = norm_stats
        return env, env_name, stats

    if env_name == "tiny":
        env = TinyLOBEnv(
            n_steps=args.n_steps,
            total_inventory=args.total_inventory,
            n_actions=args.n_actions,
            initial_price=args.initial_price,
            tick_size=args.tick_size,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            urgency_coef=args.urgency_coef,
            spread_mean=args.spread_mean,
            spread_std=args.spread_std,
            base_depth=args.light_base_depth,
            signal_rho=args.light_signal_rho,
            signal_noise=args.light_signal_noise,
            drift_strength=args.light_drift_strength,
            return_noise=args.light_return_noise,
            depth_sensitivity=args.tiny_depth_sensitivity,
            force_liquidation=args.tiny_force_liquidation,
            terminal_impact_mult=args.tiny_terminal_impact_mult,
            seed=args.seed,
        )
        stats = {"mean_spread": float(env.spread_mean), "mean_bid_vol": float(env.base_depth)}
        return env, env_name, stats

    env = LightLOBEnv(
        n_steps=args.n_steps,
        total_inventory=args.total_inventory,
        n_actions=args.n_actions,
        initial_price=args.initial_price,
        tick_size=args.tick_size,
        eta=args.eta,
        gamma_perm=args.gamma_perm,
        lambda_penalty=args.lambda_penalty,
        urgency_coef=args.urgency_coef,
        spread_mean=args.spread_mean,
        spread_std=args.spread_std,
        base_depth=args.light_base_depth,
        signal_rho=args.light_signal_rho,
        signal_noise=args.light_signal_noise,
        drift_strength=args.light_drift_strength,
        return_noise=args.light_return_noise,
        participation_impact=args.light_participation_impact,
        seed=args.seed,
    )
    stats = {"mean_spread": float(env.spread_mean), "mean_bid_vol": float(env.base_depth)}
    return env, env_name, stats


def train_tabular_q(args: argparse.Namespace) -> tuple[np.ndarray, DiscretizerSpec, dict[str, float], str, str]:
    spec = make_discretizer(args)
    rng = np.random.default_rng(args.seed)

    env, env_name, norm_stats = _build_env(args, split=args.train_split)

    table_shape = tuple(int(b) for b in spec.bins.tolist()) + (int(args.n_actions),)
    q_table = np.zeros(table_shape, dtype=np.float32)

    train_run_dir = build_run_dir(
        args.base_run_dir,
        args.train_run_name,
        overwrite=args.overwrite,
        prefix=f"tabq_{env_name}_train",
    )
    logger = RunLogger(run_dir=train_run_dir, episodes_path="", traj_path="")

    for episode in trange(args.episodes, desc="tabular-train"):
        obs, info = env.reset(seed=args.seed + episode)
        state = encode_obs(obs, spec)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0
        epsilon = epsilon_for_episode(
            episode,
            start=args.epsilon_start,
            end=args.epsilon_end,
            decay_episodes=args.epsilon_decay_episodes,
        )

        while not done:
            action = select_action(q_table, state, epsilon, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_obs(next_obs, spec)
            done = bool(terminated or truncated)

            target = float(reward)
            if not done:
                target += args.gamma * float(np.max(q_table[next_state]))
            td_error = target - float(q_table[state + (action,)])
            q_table[state + (action,)] += np.float32(args.alpha * td_error)

            step_is = _step_is(info)
            ep_reward += float(reward)
            ep_is += step_is

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": int(info.get("step", step_idx)),
                    "remaining_inventory": float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0))),
                    "action": int(action),
                    "exec_price": float(info.get("exec_price", float("nan"))),
                    "mid_price": float(info.get("mid_price", float("nan"))),
                    "reward": float(reward),
                    "is_twap": 0,
                    "executed_qty": float(info.get("executed_qty", 0.0)),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": float(step_is),
                }
            )

            state = next_state
            step_idx += 1

        remaining = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(remaining <= 1e-6),
                "mean_loss": 0.0,
                "epsilon": float(epsilon),
                "steps": step_idx,
                "strategy": f"tabular_q_train_{env_name}",
                "is_twap": 0,
            }
        )

    q_table_path = args.qtable_path or os.path.join(train_run_dir, "q_table.npz")
    save_q_table(
        out_path=q_table_path,
        q_table=q_table,
        spec=spec,
        norm_stats=norm_stats,
        n_actions=args.n_actions,
    )
    with open(os.path.join(train_run_dir, "tabular_q_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[tabular_q] Saved training artifacts to: {train_run_dir}")
    print(f"[tabular_q] Saved Q-table to: {q_table_path}")
    return q_table, spec, norm_stats, q_table_path, train_run_dir


def eval_tabular_q(
    args: argparse.Namespace,
    q_table: np.ndarray,
    spec: DiscretizerSpec,
    norm_stats: dict[str, float],
    n_actions: int,
) -> str:
    rng = np.random.default_rng(args.seed + 12345)
    args.n_actions = n_actions

    env, env_name, _ = _build_env(
        args=args,
        split=args.eval_split,
        norm_stats=(norm_stats if _resolve_env_name(args) == "abides" else None),
    )

    eval_run_dir = build_run_dir(
        args.base_run_dir,
        args.eval_run_name,
        overwrite=args.overwrite,
        prefix=f"tabq_{env_name}_eval",
    )
    logger = RunLogger(run_dir=eval_run_dir, episodes_path="", traj_path="")

    for episode in trange(args.eval_episodes, desc="tabular-eval"):
        obs, info = env.reset(seed=args.seed + 10_000 + episode)
        state = encode_obs(obs, spec)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        step_idx = 0

        while not done:
            action = greedy_action(q_table, state, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = encode_obs(next_obs, spec)
            done = bool(terminated or truncated)

            step_is = _step_is(info)
            ep_reward += float(reward)
            ep_is += step_is

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": int(info.get("step", step_idx)),
                    "remaining_inventory": float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0))),
                    "action": int(action),
                    "exec_price": float(info.get("exec_price", float("nan"))),
                    "mid_price": float(info.get("mid_price", float("nan"))),
                    "reward": float(reward),
                    "is_twap": 0,
                    "executed_qty": float(info.get("executed_qty", 0.0)),
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": float(step_is),
                }
            )

            state = next_state
            step_idx += 1

        remaining = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "completed": int(remaining <= 1e-6),
                "mean_loss": 0.0,
                "epsilon": 0.0,
                "steps": step_idx,
                "strategy": f"tabular_q_eval_{env_name}",
                "is_twap": 0,
            }
        )

    print(f"[tabular_q] Saved evaluation artifacts to: {eval_run_dir}")
    return eval_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate a tabular Q-learning agent on execution envs.")
    p.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    p.add_argument(
        "--env",
        choices=["light", "tiny", "abides"],
        default="light",
        help="Environment backend. Use --use-abides as a backward-compatible alias for abides.",
    )
    p.add_argument("--use-abides", action="store_true", help="Backward-compatible alias for --env abides.")
    p.add_argument("--npz-path", type=str, default="data/features.npz")
    p.add_argument("--train-split", choices=["train", "test", "all"], default="train")
    p.add_argument("--eval-split", choices=["train", "test", "all"], default="test")
    p.add_argument("--episodes", type=int, default=5000, help="Training episodes.")
    p.add_argument("--eval-episodes", type=int, default=200, help="Greedy evaluation episodes.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--total-inventory", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=60)
    p.add_argument("--n-actions", type=int, default=15)
    p.add_argument("--eta", type=float, default=2.5e-6)
    p.add_argument("--gamma-perm", type=float, default=2.5e-7)
    p.add_argument("--lambda-penalty", type=float, default=0.1)
    p.add_argument("--urgency-coef", type=float, default=10.0)
    p.add_argument("--initial-price", type=float, default=100.0)
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--spread-mean", type=float, default=0.05)
    p.add_argument("--spread-std", type=float, default=0.01)
    p.add_argument("--light-base-depth", type=float, default=1500.0)
    p.add_argument("--light-signal-rho", type=float, default=0.85)
    p.add_argument("--light-signal-noise", type=float, default=0.12)
    p.add_argument("--light-drift-strength", type=float, default=0.0015)
    p.add_argument("--light-return-noise", type=float, default=0.0009)
    p.add_argument("--light-participation-impact", type=float, default=0.04)
    p.add_argument("--tiny-depth-sensitivity", type=float, default=0.55)
    p.add_argument("--tiny-terminal-impact-mult", type=float, default=2.5)
    p.add_argument(
        "--tiny-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_true",
        help="Force terminal liquidation in TinyLOBEnv (default on).",
    )
    p.add_argument(
        "--tiny-no-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_false",
        help="Disable forced terminal liquidation in TinyLOBEnv.",
    )
    p.set_defaults(tiny_force_liquidation=True)

    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=int, default=20000)

    p.add_argument("--inv-bins", type=int, default=12)
    p.add_argument("--time-bins", type=int, default=12)
    p.add_argument("--mid-bins", type=int, default=1)
    p.add_argument("--spread-bins", type=int, default=1)
    p.add_argument("--vol-bins", type=int, default=1)
    p.add_argument("--obi-bins", type=int, default=1)
    p.add_argument("--mid-low", type=float, default=0.8)
    p.add_argument("--mid-high", type=float, default=1.2)
    p.add_argument("--spread-high", type=float, default=10.0)
    p.add_argument("--vol-high", type=float, default=10.0)

    p.add_argument("--base-run-dir", type=str, default="runs")
    p.add_argument("--train-run-name", type=str, default=None)
    p.add_argument("--eval-run-name", type=str, default=None)
    p.add_argument("--qtable-path", type=str, default=None, help="Path to save/load q_table.npz.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"train", "train_eval"}:
        q_table, spec, norm_stats, saved_q_path, train_run_dir = train_tabular_q(args)
    else:
        if args.qtable_path is None:
            raise ValueError("--qtable-path is required in eval mode.")
        q_table, spec, norm_stats, n_actions = load_q_table(args.qtable_path)
        if args.n_actions != n_actions:
            print(
                f"[tabular_q] Warning: overriding --n-actions={args.n_actions} "
                f"with loaded n_actions={n_actions} from q-table."
            )
        args.n_actions = n_actions
        saved_q_path = args.qtable_path
        train_run_dir = ""

    if args.mode in {"eval", "train_eval"}:
        if args.eval_run_name is None and args.mode == "train_eval":
            train_name = Path(train_run_dir).name
            args.eval_run_name = f"{train_name}_eval"
        eval_tabular_q(
            args=args,
            q_table=q_table,
            spec=spec,
            norm_stats=norm_stats,
            n_actions=args.n_actions,
        )

    if args.mode == "train":
        print(f"[tabular_q] Training complete. Q-table path: {saved_q_path}")


if __name__ == "__main__":
    main()
