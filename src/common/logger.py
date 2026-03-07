from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RunLogger:
    run_dir: str
    episodes_path: str
    traj_path: str
    _episodes_header_written: bool = False
    _traj_header_written: bool = False

    def __post_init__(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.episodes_path = os.path.join(self.run_dir, "episodes.csv")
        self.traj_path = os.path.join(self.run_dir, "trajectories.csv")

    @staticmethod
    def _append_row(path: str, row: Dict[str, Any], write_header: bool) -> bool:
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header and not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return True

    def log_episode(self, episode_row: Dict[str, Any]):
        # Expect consistent keys across calls
        self._append_row(self.episodes_path, episode_row, write_header=True)

    def log_step(self, step_row: Dict[str, Any]):
        self._append_row(self.traj_path, step_row, write_header=True)
