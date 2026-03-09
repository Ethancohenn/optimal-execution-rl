from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Allow running as either `python -m src.train_double_dqn` or `python src/train_double_dqn.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_dqn import parse_args as parse_dqn_args
from src.train_dqn import train


def parse_args():
    args = parse_dqn_args()
    args.double_dqn = True
    if args.run_name is None:
        args.run_name = datetime.now().strftime("ddqn_exec_%Y%m%d_%H%M%S")
    return args


if __name__ == "__main__":
    run_path = train(parse_args())
    print(f"Saved Double DQN artifacts to: {run_path}")
