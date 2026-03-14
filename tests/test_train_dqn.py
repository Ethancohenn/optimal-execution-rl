import argparse
from pathlib import Path

from src.train_dqn import build_run_dir, resolve_algorithm


def test_resolve_algorithm_defaults_to_dqn():
    args = argparse.Namespace(algorithm="dqn", double_dqn=False)
    algo_name, use_double = resolve_algorithm(args)
    assert algo_name == "dqn"
    assert use_double is False


def test_resolve_algorithm_uses_double_dqn_choice():
    args = argparse.Namespace(algorithm="double_dqn", double_dqn=False)
    algo_name, use_double = resolve_algorithm(args)
    assert algo_name == "double_dqn"
    assert use_double is True


def test_resolve_algorithm_alias_flag_enables_double_dqn():
    args = argparse.Namespace(algorithm="dqn", double_dqn=True)
    algo_name, use_double = resolve_algorithm(args)
    assert algo_name == "double_dqn"
    assert use_double is True


def test_build_run_dir_uses_prefix_for_auto_name():
    run_dir = build_run_dir(
        base_dir="runs",
        run_name=None,
        overwrite=False,
        default_prefix="ddqn_exec",
    )
    assert Path(run_dir).name.startswith("ddqn_exec_")
