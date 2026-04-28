from __future__ import annotations
import sys, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from environment_setting.io_config import load_config, resolve_path
from environment_setting.data_loader import load_npz_dataset
from risk_layer.wavelet_graph import WaveletGraphConfig, build_wavelet_graphs_from_returns, save_graph_cache

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
    cfg = load_config(args.config); project_root = cfg["project"]["root_dir"]
    data = load_npz_dataset(resolve_path(cfg["data"]["dataset_npz"], project_root))
    rcfg = cfg["risk"]
    wgcfg = WaveletGraphConfig(wavelet=rcfg.get("wavelet", "db2"), levels=int(rcfg.get("levels", 3)), window=int(rcfg.get("window", 252)), min_obs=int(rcfg.get("min_obs", 120)), top_m=int(rcfg.get("top_m", 5)), residual_mode=rcfg.get("residual_mode", "demean"))
    A = build_wavelet_graphs_from_returns(data.R, wgcfg)
    out = resolve_path(rcfg["graph_cache_npz"], project_root)
    save_graph_cache(out, data.dates, data.instruments, A)
    print(f"[OK] saved graph cache: {out}, A={A.shape}")
if __name__ == "__main__": main()
