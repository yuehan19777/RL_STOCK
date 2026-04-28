from __future__ import annotations
import sys, argparse, subprocess
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    print("[RUN]", " ".join(cmd)); subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--with-ppo", action="store_true"); args = ap.parse_args()
    py = sys.executable; cfg = args.config
    run([py, str(PROJECT_ROOT/"run_main"/"precompute_wavelet_graphs.py"), "--config", cfg])
    run([py, str(PROJECT_ROOT/"run_main"/"train_signal.py"), "--config", cfg])
    run([py, str(PROJECT_ROOT/"run_main"/"backtest_eval.py"), "--config", cfg, "--mode", "fixed"])
    if args.with_ppo:
        run([py, str(PROJECT_ROOT/"run_main"/"train_ppo_controller.py"), "--config", cfg])
        run([py, str(PROJECT_ROOT/"run_main"/"backtest_eval.py"), "--config", cfg, "--mode", "ppo"])
if __name__ == "__main__": main()
