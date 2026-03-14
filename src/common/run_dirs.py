from __future__ import annotations

import os
import shutil
import stat
from datetime import datetime


def _make_writable(path: str) -> None:
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | stat.S_IWRITE)
    except OSError:
        pass


def _clear_readonly_tree(root: str) -> None:
    if not os.path.exists(root):
        return

    _make_writable(root)
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            _make_writable(os.path.join(dirpath, dirname))
        for filename in filenames:
            _make_writable(os.path.join(dirpath, filename))


def remove_run_dir(run_dir: str) -> None:
    _clear_readonly_tree(run_dir)
    shutil.rmtree(run_dir)


def build_run_dir(base_dir: str, run_name: str | None, overwrite: bool, default_prefix: str) -> str:
    if run_name is None:
        run_name = datetime.now().strftime(f"{default_prefix}_%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, run_name)
    if os.path.exists(run_dir):
        if overwrite:
            try:
                remove_run_dir(run_dir)
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
