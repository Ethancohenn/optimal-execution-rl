"""
Optuna hyperparameter tuning for the DQN agent on optimal execution.

Usage
-----
  python scripts/tune_dqn.py --use-abides --n-trials 50 --episodes 200
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_dqn import train


def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    """Optuna objective function."""
    args = argparse.Namespace(**vars(base_args))
    args.run_name = f"optuna_trial_{trial.number}"
    args.overwrite = True
    args.save_model = False # No need to save models for every single trial initially

    # --- Search Space Definition ---
    # User specifically requested prioritizing episodes, urgency-coef, and lambda-penalty
    args.episodes = trial.suggest_int("episodes", 100, 1000, step=100)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    args.urgency_coef = trial.suggest_float("urgency_coef", 0.0, 100.0)
    args.lambda_penalty = trial.suggest_float("lambda_penalty", 0.1, 10.0)
    args.n_actions = trial.suggest_int("n_actions", 5, 25, step=5)
    
    # We still tune some core DQN params to ensure the model converges nicely at all
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.gamma = trial.suggest_float("gamma", 0.90, 0.999)

    # Convert lists to tuple to be hashable if needed, though argparse expects list for hidden_dims
    hidden_dim_choice = trial.suggest_categorical("hidden_dims", ["64_64", "128_128", "256_256"])
    args.hidden_dims = [int(x) for x in hidden_dim_choice.split("_")]

    # --- Run Training ---
    try:
        run_dir = train(args)
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # --- Calculate Metric ---
    # We want to minimize the Implementation Shortfall (IS) of the *last N* episodes.
    # We must penalize agents that don't complete their execution (creating artificially low IS).
    episodes_path = os.path.join(run_dir, "episodes.csv")
    if not os.path.exists(episodes_path):
        return float("inf")
    
    df = pd.read_csv(episodes_path)
    if len(df) == 0:
        return float("inf")
    
    # Take the last 20% of episodes as the evaluation window
    eval_window = max(1, int(0.2 * args.episodes))
    df_eval = df.tail(eval_window)
    
    avg_is = df_eval["episode_is"].mean()
    completion_rate = df_eval["completed"].mean()
    
    # Massive penalty per missing % completion to prevent "lazy agent" local minima
    penalty = (1.0 - completion_rate) * 10000.0
    
    return float(avg_is + penalty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DQN hyperparameters using Optuna.")
    # Generic train arguments that we'll keep constant
    parser.add_argument("--episodes", type=int, default=200, help="Fewer episodes for tuning")
    parser.add_argument("--seed", type=int, default=42)
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
    
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--double-dqn", action="store_true", default=True)
    parser.add_argument("--smoothness-coef", type=float, default=0.01)
    parser.add_argument("--target-update-interval", type=int, default=50)
    
    parser.add_argument("--base-run-dir", type=str, default="runs/optuna")
    parser.add_argument("--use-abides", action="store_true")
    parser.add_argument("--npz-path", type=str, default="data/features.npz")
    parser.add_argument("--split", type=str, default="train")

    # Optuna specific
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", type=str, default="dqn_tuning")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.base_run_dir, exist_ok=True)
    
    # Optimize to MINIMIZE implementation shortfall
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    print(f"Starting Optuna search with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("\n" + "="*50)
    print("Optimization finished!")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Avg Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50 + "\n")
    
    # Save results
    results_df = study.trials_dataframe()
    out_path = os.path.join(args.base_run_dir, f"{args.study_name}_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved full trial results to: {out_path}")


if __name__ == "__main__":
    main()
