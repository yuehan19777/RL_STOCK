from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment_setting.data_loader import load_npz_dataset, split_indices_from_cfg
from controller_ppo.meta_env import MetaPortfolioEnv
from controller_ppo.ppo import train_ppo


def load_cfg(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--train_on", type=str, default="train", choices=["train", "train_valid"])
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg = load_cfg(cfg_path)

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    dataset_npz = cfg["data"]["dataset_npz"]
    graph_npz = cfg["risk"]["graph_cache_npz"]
    pred_npz = cfg["signal"]["predictions_npz"]

    data = load_npz_dataset(dataset_npz)

    dates = data.dates
    R = data.R

    tr_idx, va_idx, te_idx = split_indices_from_cfg(dates, cfg)

    if args.train_on == "train":
        train_idx = tr_idx
    else:
        train_idx = np.concatenate([tr_idx, va_idx])

    g = np.load(graph_npz, allow_pickle=True)
    A_scales = g["A_scales"].astype(np.float32)

    p = np.load(pred_npz, allow_pickle=True)
    mu_pred = p["pred"].astype(np.float32)

    env = MetaPortfolioEnv(
        R=R,
        mu_pred=mu_pred,
        A_scales=A_scales,
        dates=dates,
        idx=train_idx,
        cfg=cfg,
    )

    result = train_ppo(env, cfg)

    model_path = os.path.join(out_dir, "ppo_controller.pt")
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
            "cfg": cfg,
        },
        model_path,
    )

    hist_path = os.path.join(out_dir, "ppo_training_history.csv")
    pd.DataFrame(result.history).to_csv(hist_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved PPO model to {model_path}")
    print(f"[OK] saved PPO history to {hist_path}")


if __name__ == "__main__":
    main()