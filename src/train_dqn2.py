from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution_infra import EnvConfig, ExecutionEnv, LightLOBEnv, TinyLOBEnv
from execution_infra.abides_replay_env import AbidesReplayEnv
from src.agents.dqn import DQNAgent
from src.common.logger import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN/Double-DQN with an IS-aligned objective "
            "(reward = - qty * (mid - exec))."
        )
    )

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--total-inventory", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--initial-price", type=float, default=100.0)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--volatility", type=float, default=0.002)
    parser.add_argument("--spread-mean", type=float, default=0.05)
    parser.add_argument("--spread-std", type=float, default=0.01)
    parser.add_argument("--lob-depth-mean", type=int, default=500)
    parser.add_argument("--lob-depth-std", type=int, default=100)
    parser.add_argument("--n-lob-levels", type=int, default=3)
    parser.add_argument("--eta", type=float, default=2.5e-6)
    parser.add_argument("--gamma-perm", type=float, default=2.5e-7)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--urgency-coef", type=float, default=0.0)

    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "double_dqn"],
        help="Training algorithm. Use 'double_dqn' for Double DQN targets.",
    )
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Alias for --algorithm double_dqn.",
    )
    parser.add_argument("--smoothness-coef", type=float, default=0.01)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)

    parser.add_argument(
        "--leftover-penalty-coef",
        type=float,
        default=0.01,
        help=(
            "Terminal penalty coefficient on leftover notional in the aligned objective. "
            "Applied as: reward -= coef * leftover_shares * benchmark_price."
        ),
    )
    parser.add_argument(
        "--hold-penalty-coef",
        type=float,
        default=0.01,
        help=(
            "Dense per-step inventory holding penalty coefficient. "
            "Applied as: reward -= coef * (inventory_remaining / total_inventory) * benchmark_price."
        ),
    )

    parser.add_argument("--base-run-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete existing run directory before writing."
    )
    parser.add_argument("--save-model", action="store_true", help="Save policy network to run_dir/policy_net.pt")
    parser.add_argument(
        "--env",
        type=str,
        default="execution",
        choices=["execution", "light", "tiny", "abides"],
        help=(
            "Environment backend: 'execution' (old synthetic env), "
            "'light' (lightweight learnable LOB), "
            "'tiny' (extra-fast sanity-check LOB), or 'abides' (replay data)."
        ),
    )
    parser.add_argument(
        "--use-abides",
        action="store_true",
        help="Backward-compatible alias for --env abides.",
    )
    parser.add_argument(
        "--npz-path",
        type=str,
        default="data/features.npz",
        help="Path to features.npz produced by the feature extraction pipeline.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Train/test split of the data file (first 80%% or last 20%%). Default: train.",
    )

    parser.add_argument("--light-base-depth", type=float, default=1500.0)
    parser.add_argument("--light-signal-rho", type=float, default=0.85)
    parser.add_argument("--light-signal-noise", type=float, default=0.12)
    parser.add_argument("--light-drift-strength", type=float, default=0.0015)
    parser.add_argument("--light-return-noise", type=float, default=0.0009)
    parser.add_argument("--light-participation-impact", type=float, default=0.04)
    parser.add_argument("--tiny-depth-sensitivity", type=float, default=0.55)
    parser.add_argument("--tiny-terminal-impact-mult", type=float, default=2.5)
    parser.add_argument(
        "--tiny-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_true",
        help="Force terminal liquidation in TinyLOBEnv (default on).",
    )
    parser.add_argument(
        "--tiny-no-force-liquidation",
        dest="tiny_force_liquidation",
        action="store_false",
        help="Disable forced terminal liquidation in TinyLOBEnv.",
    )
    parser.set_defaults(tiny_force_liquidation=True)
    return parser.parse_args()


def resolve_algorithm(args: argparse.Namespace) -> tuple[str, bool]:
    use_double_dqn = bool(args.double_dqn or args.algorithm == "double_dqn")
    algo_name = "double_dqn" if use_double_dqn else "dqn"
    return algo_name, use_double_dqn


def build_run_dir(
    base_dir: str,
    run_name: str | None,
    overwrite: bool,
    default_prefix: str,
) -> str:
    if run_name is None:
        run_name = datetime.now().strftime(f"{default_prefix}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            try:
                shutil.rmtree(run_dir)
            except PermissionError as exc:
                raise PermissionError(
                    f"Cannot overwrite run directory '{run_dir}'. "
                    "Close files under this folder (editor/Explorer) or use a different --run-name."
                ) from exc
        else:
            raise ValueError(
                f"Run directory already exists: {run_dir}. "
                "Use --overwrite or choose a new --run-name."
            )
    return run_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_step_is(info: dict) -> float:
    executed_qty = float(info.get("executed_qty", 0.0))
    if executed_qty <= 0:
        return 0.0

    exec_price = float(info.get("exec_price", float("nan")))
    mid_price = float(info.get("mid_price", float("nan")))
    if np.isnan(exec_price) or np.isnan(mid_price):
        return 0.0

    return executed_qty * (mid_price - exec_price)


def objective_reward_from_info(
    info: dict,
    done: bool,
    leftover_penalty_coef: float,
    hold_penalty_coef: float,
    total_inventory: float,
) -> tuple[float, float, float, float]:
    """
    Returns (objective_reward, step_is, hold_penalty, leftover_penalty).
    Objective is aligned with comparison IS metric:
      objective_reward = -step_is - hold_penalty - leftover_penalty
      step_is = qty * (mid - exec)
    """
    step_is = compute_step_is(info)
    hold_penalty = 0.0
    leftover_penalty = 0.0

    if hold_penalty_coef > 0.0:
        inv_left = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        benchmark_price = float(info.get("benchmark_price", 1.0))
        inv_frac = inv_left / max(float(total_inventory), 1.0)
        hold_penalty = float(hold_penalty_coef * inv_frac * max(benchmark_price, 1.0))

    if done and leftover_penalty_coef > 0.0:
        leftover = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))
        benchmark_price = float(info.get("benchmark_price", 1.0))
        if leftover > 0.0:
            leftover_penalty = float(leftover_penalty_coef * leftover * max(benchmark_price, 1.0))

    objective_reward = -step_is - hold_penalty - leftover_penalty
    return float(objective_reward), float(step_is), float(hold_penalty), float(leftover_penalty)


def _resolve_env_name(args: argparse.Namespace) -> str:
    return "abides" if args.use_abides else str(args.env)


def _build_env(args: argparse.Namespace) -> tuple[object, str, dict[str, float] | None]:
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
            split=args.split,
        )
        print(f"[train_dqn2] Using AbidesReplayEnv  npz={args.npz_path} (split={args.split})")
        norm_stats = {"mean_spread": env._mean_spread, "mean_bid_vol": env._mean_bid_vol}
        return env, env_name, norm_stats

    if env_name == "execution":
        cfg = EnvConfig(
            total_inventory=args.total_inventory,
            n_steps=args.n_steps,
            n_actions=args.n_actions,
            initial_price=args.initial_price,
            tick_size=args.tick_size,
            volatility=args.volatility,
            spread_mean=args.spread_mean,
            spread_std=args.spread_std,
            lob_depth_mean=args.lob_depth_mean,
            lob_depth_std=args.lob_depth_std,
            n_lob_levels=args.n_lob_levels,
            eta=args.eta,
            gamma_perm=args.gamma_perm,
            lambda_penalty=args.lambda_penalty,
            seed=args.seed,
        )
        env = ExecutionEnv(config=cfg)
        print("[train_dqn2] Using synthetic ExecutionEnv")
        return env, env_name, None

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
        print("[train_dqn2] Using TinyLOBEnv")
        return env, env_name, None

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
    print("[train_dqn2] Using LightLOBEnv")
    return env, env_name, None


def train(args: argparse.Namespace) -> str:
    set_global_seed(args.seed)
    algo_name, use_double_dqn = resolve_algorithm(args)
    if args.leftover_penalty_coef <= 0.0 and args.hold_penalty_coef <= 0.0:
        print(
            "[train_dqn2] Warning: leftover_penalty_coef <= 0 and hold_penalty_coef <= 0. "
            "The aligned objective can collapse to 'never trade' (0 IS, 0 completion)."
        )
    print(
        f"[train_dqn2] Algorithm={algo_name} (double_dqn={use_double_dqn}) "
        f"| objective=is_aligned | hold_penalty_coef={args.hold_penalty_coef} "
        f"| leftover_penalty_coef={args.leftover_penalty_coef}"
    )

    env, env_name, norm_stats = _build_env(args)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        double_dqn=use_double_dqn,
        smoothness_coef=args.smoothness_coef,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
    )

    run_prefix = f"{'ddqn2' if use_double_dqn else 'dqn2'}_is_{env_name}"
    run_dir = build_run_dir(
        args.base_run_dir,
        args.run_name,
        overwrite=args.overwrite,
        default_prefix=run_prefix,
    )
    logger = RunLogger(run_dir=run_dir, episodes_path="", traj_path="")

    cfg_path = os.path.join(run_dir, "objective_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": "is_aligned",
                "env": env_name,
                "hold_penalty_coef": float(args.hold_penalty_coef),
                "leftover_penalty_coef": float(args.leftover_penalty_coef),
                "eta": float(args.eta),
                "gamma_perm": float(args.gamma_perm),
                "lambda_penalty_env": float(args.lambda_penalty),
                "urgency_coef_env": float(args.urgency_coef),
            },
            f,
            indent=2,
        )

    for episode in trange(args.episodes, desc="train_dqn2"):
        state, info = env.reset(seed=args.seed + episode)
        done = False
        ep_reward = 0.0
        ep_is = 0.0
        ep_env_reward = 0.0
        ep_hold_penalty = 0.0
        ep_leftover_penalty = 0.0
        ep_loss = 0.0
        loss_count = 0
        prev_action = 0
        step_idx = 0

        while not done:
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            reward, step_is, hold_penalty, leftover_penalty = objective_reward_from_info(
                info=info,
                done=done,
                leftover_penalty_coef=args.leftover_penalty_coef,
                hold_penalty_coef=args.hold_penalty_coef,
                total_inventory=float(args.total_inventory),
            )

            agent.store_transition(state, action, reward, next_state, done, prev_action)
            metrics = agent.train_step()
            if metrics is not None:
                ep_loss += metrics["loss"]
                loss_count += 1

            executed_qty = float(info.get("executed_qty", 0.0))
            exec_price = float(info.get("exec_price", float("nan")))
            mid_price = float(info.get("mid_price", float("nan")))
            t_val = int(info.get("t", info.get("step", step_idx)))
            remaining = float(info.get("inventory_remaining", info.get("remaining_inventory", 0.0)))

            logger.log_step(
                {
                    "episode": episode,
                    "step": step_idx,
                    "t": t_val,
                    "remaining_inventory": remaining,
                    "action": int(action),
                    "exec_price": exec_price,
                    "mid_price": mid_price,
                    "reward": float(reward),
                    "env_reward": float(env_reward),
                    "is_twap": 0,
                    "executed_qty": executed_qty,
                    "forced_qty": float(info.get("forced_qty", 0.0)),
                    "step_is": float(step_is),
                    "hold_penalty": float(hold_penalty),
                    "leftover_penalty": float(leftover_penalty),
                }
            )

            ep_reward += float(reward)
            ep_is += float(step_is)
            ep_env_reward += float(env_reward)
            ep_hold_penalty += float(hold_penalty)
            ep_leftover_penalty += float(leftover_penalty)
            prev_action = int(action)
            state = next_state
            step_idx += 1

        logger.log_episode(
            {
                "episode": episode,
                "episode_reward": float(ep_reward),
                "episode_is": float(ep_is),
                "episode_env_reward": float(ep_env_reward),
                "episode_hold_penalty": float(ep_hold_penalty),
                "episode_leftover_penalty": float(ep_leftover_penalty),
                "completed": int(float(info.get("inventory_remaining", 1.0)) <= 0),
                "mean_loss": float(ep_loss / max(1, loss_count)),
                "epsilon": float(agent.epsilon()),
                "steps": step_idx,
                "strategy": "dqn2_is_aligned" if not use_double_dqn else "ddqn2_is_aligned",
                "is_twap": 0,
            }
        )

    if args.save_model:
        model_path = os.path.join(run_dir, "policy_net.pt")
        torch.save(agent.policy_net.state_dict(), model_path)
        if norm_stats is not None:
            with open(os.path.join(run_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
                json.dump(norm_stats, f, indent=2)

    return run_dir


if __name__ == "__main__":
    args = parse_args()
    algo_name, _ = resolve_algorithm(args)
    run_path = train(args)
    print(f"Saved {algo_name} dqn2 artifacts to: {run_path}")
