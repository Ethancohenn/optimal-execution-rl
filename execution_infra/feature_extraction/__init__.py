"""
Feature extraction package for ABIDES simulation logs.

Extracts market microstructure features from bz2-compressed log files
and produces training-ready numpy arrays / DataFrames.

Quick usage
-----------
    from execution_infra.feature_extraction import extract_features

    # Returns (DataFrame, dict of numpy arrays)
    df, arrays = extract_features("abides/log/my_sim_run", symbol="IBM")

    # Save to .npz
    extract_features("abides/log/my_sim_run", output_path="data/features.npz")
"""

from execution_infra.feature_extraction.pipeline import extract_features, FeatureConfig

__all__ = ["extract_features", "FeatureConfig"]
