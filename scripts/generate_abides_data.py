import os
import sys
import subprocess
import argparse
from multiprocessing import Process
from pathlib import Path
from execution_infra.feature_extraction.pipeline import extract_features, FeatureConfig
import pandas as pd
import numpy as np
import shutil

def run_simulation(datestr: str, seed: int, log_dir: str):
    print(f"[{datestr}] Starting ABIDES simulation...")
    # We run full trading hours from 9:30 to 16:00
    cmd = [
        sys.executable, "-u", "abides/abides.py",
        "-c", "rmsc03",
        "-t", "ABM",
        "-d", datestr,
        "-s", str(seed),
        "-l", log_dir,
        "--start-time", "09:30:00",
        "--end-time", "16:00:00",
        "-e", "-p", "0.05"  # POV Execution Agent
    ]
    # Use Popen to stream stdout and stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"[{datestr}] {line}", end="")
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    print(f"[{datestr}] Simulation finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-days", type=int, default=5, help="Number of days to simulate")
    parser.add_argument("--start-date", type=str, default="20200601", help="Start date (YYYYMMDD)")
    parser.add_argument("--output", type=str, default="data/features_large.npz", help="Output Path")
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    # Generate date strings
    start_date = pd.to_datetime(args.start_date)
    # Filter out weekends to be safe
    dates = pd.bdate_range(start=start_date, periods=args.num_days).strftime('%Y%m%d').tolist()
    
    processes = []
    log_dirs = []
    
    # Step 1: Run simulations in parallel
    for i, datestr in enumerate(dates):
        # We need a unique log directory name for each simulation run
        log_dir = f"multiday_sim_{datestr}"
        log_dirs.append(log_dir)
        
        # We must change the random seed for each day so we don't get identical market order books
        seed = args.base_seed + i
        
        p = Process(target=run_simulation, args=(datestr, seed, log_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print("Error: A simulation process failed!")
            return

    # Step 2: Extract features for each day sequentially
    print("\nSimulations complete. Starting feature extraction...")
    all_dfs = []
    
    cfg = FeatureConfig(
        symbol="ABM",
        resample_freq="1s",
        total_inventory=1200000, 
    )

    for log_dir in log_dirs:
        # Fixed path to look in /log/ instead of /abides/log/
        full_log_path = Path("log") / log_dir
        if not full_log_path.exists():
            print(f"Warning: {full_log_path} does not exist. Skipping.")
            continue
            
        print(f"Extracting features for {log_dir}...")
        df, _ = extract_features(full_log_path, config=cfg)
        all_dfs.append(df)
        
        # Cleanup massive bz2 logs to save disk space
        shutil.rmtree(full_log_path)

    if not all_dfs:
        print("Error: No dataframes generated.")
        return

    # Step 3: Concatenate and Save
    print("\nConcatenating days...")
    # `pd.concat` places the days in order, keeping the time series continuous
    final_df = pd.concat(all_dfs, axis=0)
    
    # Convert to npz format expected by AbidesReplayEnv
    feature_names = np.array(final_df.columns.tolist())
    timestamps = np.array(final_df.index.strftime("%Y-%m-%d %H:%M:%S.%f"))
    features = final_df.values.astype(np.float32)

    arrays = {
        "features": features,
        "feature_names": feature_names,
        "timestamps": timestamps,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)
    
    print("\n" + "=" * 60)
    print(f"Successfully saved massive dataset to {output_path}")
    print(f"Total Rows: {final_df.shape[0]} (Old dataset: ~7000 rows)")
    print(f"Total Columns: {final_df.shape[1]}")
    print("=" * 60)

if __name__ == "__main__":
    main()
