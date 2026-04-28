from __future__ import annotations
"""
Optional script: build Alpha360-style NPZ from Qlib CN data.
If you already have WDG_RL/data/processed/*.npz, you can skip this script.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from environment_setting.io_config import load_config, resolve_path, ensure_dir

def init_qlib(provider_uri: str):
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=provider_uri, region=REG_CN)

def list_instruments_by_file(provider_uri: str, universe: str, start: str, end: str):
    inst_path = os.path.join(provider_uri, "instruments", f"{universe}.txt")
    if not os.path.exists(inst_path): raise FileNotFoundError(inst_path)
    s_req, e_req = pd.Timestamp(start), pd.Timestamp(end); out = []
    with open(inst_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if not p: continue
            inst = p[0]
            if len(p) >= 3:
                s = pd.Timestamp(p[1]); e = pd.Timestamp(p[2])
                if e < s_req or s > e_req: continue
            out.append(inst)
    return list(dict.fromkeys(out))

def fetch_field_panel(instruments, fields, start, end):
    from qlib.data import D

    df = D.features(
        instruments,
        [f"${f}" for f in fields],
        start_time=start,
        end_time=end
    ).reset_index()

    df["datetime"] = pd.to_datetime(df["datetime"])

    panels = []
    dates = None

    for f in fields:
        piv = (
            df.pivot(index="datetime", columns="instrument", values=f"${f}")
            .sort_index()
            .reindex(columns=instruments)
            .ffill()
            .bfill()
        )

        if dates is None:
            dates = piv.index

        panels.append(piv.values.astype(np.float32))

    if dates is None:
        raise ValueError("No dates fetched from Qlib. Please check instruments, fields, start, end.")

    return dates, panels
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default=None); args = ap.parse_args()
    cfg = load_config(args.config); dcfg = cfg["data"]; provider = dcfg["qlib_provider_uri"]; init_qlib(provider)
    inst_all = list_instruments_by_file(provider, dcfg["universe"], dcfg["start_date"], dcfg["end_date"])
    _, panels_v = fetch_field_panel(inst_all, ["volume"], dcfg["train_start_for_topn"], dcfg["train_end_for_topn"])
    order = np.argsort(np.nanmean(panels_v[0], axis=0))[::-1][:int(dcfg["n_stocks"])]
    instruments = [inst_all[i] for i in order]; print("[Dataset] selected instruments:", instruments)
    fields = dcfg.get("fields", ["open", "high", "low", "close", "vwap", "volume"]); dates, panels = fetch_field_panel(instruments, fields, dcfg["start_date"], dcfg["end_date"])
    L = int(dcfg.get("alpha360_lookback", 60)); Traw, N = panels[0].shape; samples = []; out_dates = []
    for t in range(L-1, Traw-1):
        feat = [arr[t-L+1:t+1].T for arr in panels]
        samples.append(np.concatenate(feat, axis=1)); out_dates.append(pd.Timestamp(dates[t]).strftime("%Y-%m-%d"))
    X = np.stack(samples).astype(np.float32); close = panels[fields.index("close")]
    R = (close[L:Traw] / (close[L-1:Traw-1] + 1e-12) - 1.0).astype(np.float32)
    feature_names = [f"{f}_lag{lag}" for f in fields for lag in range(L, 0, -1)]
    out = args.out or resolve_path(dcfg["dataset_npz"], cfg["project"]["root_dir"]); ensure_dir(os.path.dirname(out))
    np.savez_compressed(out, X=X, R=R, dates=np.array(out_dates, dtype=object), instruments=np.array(instruments, dtype=object), feature_names=np.array(feature_names, dtype=object))
    print(f"[OK] saved {out}: X={X.shape}, R={R.shape}")
    
if __name__ == "__main__": main()
