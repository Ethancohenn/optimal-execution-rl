# Optimal Execution with Reinforcement Learning

A CS234 project studying optimal trade execution as a sequential decision problem. We train RL agents (DQN, Double DQN) to liquidate a large inventory while minimising market impact, and compare them against TWAP/VWAP baselines.

## Repository Structure

```
optimal-execution-rl/
├── abides/            ← ABIDES multi-agent market simulator (Python 3.6)
│   ├── config/        ← Simulation configurations (execution.py, qlearning.py, …)
│   ├── agent/         ← Agent implementations (TWAP, VWAP, POV, …)
│   ├── scripts/       ← Shell scripts to run simulations
│   └── requirements.txt
├── execution_infra/   ← Our RL environment (Python 3.11)
│   ├── config.py      ← Centralised hyperparameters (EnvConfig)
│   ├── market_sim.py  ← Lightweight LOB simulator with market impact
│   ├── execution_env.py ← gymnasium.Env wrapper
│   ├── feature_extraction/  ← ABIDES log → training features pipeline
│   │   ├── parsers.py       ← bz2 log readers (exchange, fundamental, exec agent)
│   │   ├── features.py      ← Feature engineering (orderbook, trade, derived)
│   │   └── pipeline.py      ← End-to-end orchestrator + CLI
│   └── README.md      ← How to build RL agents on this env
├── data/              ← Extracted feature files (.npz)
└── README.md          ← This file
```

---

## 1 · Setting Up ABIDES (Data-Generation Only)

ABIDES requires **Python 3.6**. We use it solely to generate realistic LOB simulation data — all RL code runs in a separate, modern Python environment.

### macOS

```bash
# 1. Install Miniconda (skip if you have conda)
brew install --cask miniconda
conda init zsh   # or conda init bash

# 2. Create the ABIDES environment(if you are on Apple Silicon Mac)
CONDA_SUBDIR=osx-64 conda create -n abides_sim python=3.6 -y
conda activate abides_sim
conda config --env --set subdir osx-64

# 3. Install dependencies
cd abides
pip install -r requirements.txt

# 4. Run a sample execution simulation
cd ..                           # back to repo root
python abides/abides.py -c execution -t ABM -d 20200101 -s 123456789 -e
```

### Windows

#### Option A — Miniconda (recommended)

```powershell
# 1. Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# 2. Open Anaconda Prompt
conda create -n abides_sim python=3.6 -y
conda activate abides_sim

# 3. Install dependencies
cd abides
pip install -r requirements.txt

# 4. Run a sample simulation
cd ..
python abides\abides.py -c execution -t ABM -d 20200101 -s 123456789 -e
```

#### Option B — WSL (Windows Subsystem for Linux)

If you encounter issues with Python 3.6 on native Windows, use WSL:

```bash
# Inside WSL (Ubuntu)
sudo apt update && sudo apt install -y wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart shell

conda create -n abides_sim python=3.6 -y
conda activate abides_sim
cd /mnt/c/Users/<YOUR_USERNAME>/path/to/optimal-execution-rl/abides
pip install -r requirements.txt
```

### Troubleshooting ABIDES

| Issue | Fix |
|-------|-----|
| `python 3.6 not found` | Use `conda create -n abides_sim python=3.6.13` or install via pyenv |
| `ModuleNotFoundError: Kernel` | Make sure you run `abides.py` from the `abides/` directory, or use the full path |
| Slow simulation | Reduce `num_noise` in `config/execution.py` (default 5000) |

---

## 2 · Setting Up the RL Environment

The RL environment uses **Python 3.11** and is completely independent of ABIDES.

### macOS & Windows (same steps)

```bash
# 1. Create a new conda environment
conda create -n cs234_rl python=3.11 -y
conda activate cs234_rl

# 2. Install dependencies
pip install gymnasium numpy torch matplotlib pandas tqdm

# 3. Verify the environment works
cd /path/to/optimal-execution-rl
python -c "
from execution_infra import ExecutionEnv
env = ExecutionEnv()
obs, info = env.reset()
print('Observation:', obs)
print('Action space:', env.action_space)
for i in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    print(f'  step {i+1}: reward={r:.4f}, inv={info[\"inventory_remaining\"]}')
print('Environment OK!')
"
```

### Quick Start — Training an Agent

```python
from execution_infra import ExecutionEnv, EnvConfig

# Create environment with custom config
config = EnvConfig(total_inventory=1000, n_steps=60, lambda_penalty=1.0)
env = ExecutionEnv(config)

obs, info = env.reset(seed=42)
total_reward = 0

for step in range(config.n_steps):
    action = env.action_space.sample()  # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward:.2f}")
print(f"Completion: {info['completion_rate']:.1%}")
print(f"Shortfall:  {info['implementation_shortfall']:.2f}")
```

---

## 3 · Generating ABIDES Simulation Data

Use the ABIDES conda environment (`abides_sim`) to produce LOB order-book logs:

```bash
conda activate abides_sim
cd abides

# Run the execution config (TWAP agent in a realistic LOB)
python abides.py -c execution -t ABM -d 20200101 -s 123456789 -e -l my_run

# Output lands in  abides/log/my_run/
#   -> ExchangeAgent0.bz2   (order book snapshots)
#   -> ORDERBOOK_ABM_FULL.bz2  (wide book)
```

Key flags:

| Flag | Meaning |
|------|---------|
| `-c execution` | Use the `config/execution.py` configuration |
| `-t ABM` | Ticker symbol |
| `-d 20200101` | Simulated historical date |
| `-s 123456789` | Random seed (for reproducibility) |
| `-e` | Enable the execution agent (TWAP by default) |
| `-l my_run` | Name of the log directory |

You can swap the execution agent in `config/execution.py` (TWAP, VWAP, or POV are all already implemented — just uncomment the one you want).

---

## 4 · Feature Extraction Pipeline

The feature extraction pipeline converts raw ABIDES `.bz2` log files into a training-ready `.npz` dataset. It parses the exchange event stream (best bid/ask updates, executed trades), the oracle fundamental value series, and any execution agent logs, then computes 17 market microstructure features — including mid-price, spread, order book imbalance, traded volume, trade intensity, rolling volatility, VWAP, and benchmark price — all resampled onto a uniform 1-second time grid.

### CLI Usage

```bash
conda activate cs234_rl

python3 -m execution_infra.feature_extraction.pipeline \
    --log-dir abides/log/my_sim_run \
    --symbol IBM \
    --freq 1s \
    --output data/features.npz
```

### Python Usage

```python
from execution_infra.feature_extraction import extract_features

# Returns a DataFrame and a dict of numpy arrays
df, arrays = extract_features("abides/log/my_sim_run")

# Or save directly to .npz
df, arrays = extract_features("abides/log/my_sim_run", output_path="data/features.npz")

# Load later
import numpy as np
data = np.load("data/features.npz")
X = data["features"]          # shape (7200, 17), float32
names = data["feature_names"]  # array of 17 column name strings
```

### Output Features (17 columns)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `best_bid` | Exchange | Best bid price ($) |
| 2 | `best_ask` | Exchange | Best ask price ($) |
| 3 | `mid_price` | Exchange | (bid + ask) / 2 |
| 4 | `spread` | Exchange | ask − bid ($) |
| 5 | `bid_vol_1` | Exchange | Volume at best bid |
| 6 | `ask_vol_1` | Exchange | Volume at best ask |
| 7 | `order_book_imbalance` | Exchange | bid_vol / (bid + ask) |
| 8 | `last_trade_price` | Exchange | Most recent fill price ($) |
| 9 | `traded_volume` | Exchange | Shares traded per second |
| 10 | `trade_intensity` | Exchange | Number of trades per second |
| 11 | `remaining_inventory` | Exec Agent | Shares left to sell |
| 12 | `executed_volume` | Exec Agent | Cumulative shares sold |
| 13 | `last_execution_price` | Exec Agent | Last fill price ($) |
| 14 | `time_remaining` | Derived | 1.0 → 0.0 over trading session |
| 15 | `volatility` | Derived | Rolling std of mid-price log-returns |
| 16 | `vwap` | Derived | Cumulative VWAP |
| 17 | `benchmark_price` | Fundamental | Oracle fundamental value ($) |

---

## 5 · Key Concepts

### State Space ($s_t$)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | Inventory remaining | [0, 1] | Normalised: current / initial |
| 1 | Time remaining | [1 → 0] | Normalised: 1 at start, 0 at deadline |
| 2 | Spread | [0, ∞) | Best ask − best bid ($) |
| 3 | LOB volume | [0, 1] | Normalised sum of top-3 bid+ask depth |

### Action Space

| Action | Fraction sold |
|--------|--------------|
| 0 | 0% (wait) |
| 1 | 25% of remaining |
| 2 | 50% of remaining |
| 3 | 75% of remaining |
| 4 | 100% of remaining |

### Market Impact

When the agent sells $q$ shares:

$$P_t = P_{\text{mid}} \cdot \big(1 - \eta \cdot q_t - \gamma \cdot \text{CumulativeVol}\big)$$

- $\eta$ = temporary impact (default `2.5e-6`)
- $\gamma$ = permanent impact (default `2.5e-7`)

### Reward Function

The agent's objective is to minimize execution slippage relative to the benchmark price. At each step $t$, the reward is calculated as:

$$r_t = -(P_{\text{exec}} - P_{\text{benchmark}}) \cdot q_t$$

Where:
* $P_{\text{exec}}$: The price at which the shares were filled.
* $P_{\text{benchmark}}$: The target reference price (e.g., Arrival Price or VWAP).
* $q_t$ : The quantity traded at step $t$.
---

## 6 · Implementing a DQN Agent

Create a file (e.g. `agents/dqn.py`) with the following structure. The environment is already gymnasium-compatible, so the training loop is standard

## Team
Mingyang Li, Haotian Cui, Ethan Cohen
CS234 — Reinforcement Learning, Stanford University
