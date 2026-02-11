Install all required packages using pip:

```bash
pip install -r requirements.txt
```

## Two-environment workflow (recommended)

Use one environment for ABIDES (legacy Python) and one for the rest of the project (modern Python).

### 1) ABIDES environment (Python 3.6)

```bash
cd abides
conda create -n CS234Proj python=3.6
conda activate CS234Proj
pip install -r requirements.txt
python abides.py -c rmsc03
```

### 2) Main project environment (Python 3.11)

```bash
cd ..
conda create -n exec-rl python=3.11
conda activate exec-rl
pip install -r requirements.txt
pip install gymnasium numpy pandas matplotlib seaborn torch
```

### Which env to use?

- Use `CS234Proj` when running ABIDES simulations from `abides/`.
- Use `exec-rl` for RL code, tests, baselines, metrics, and plots in the root project.