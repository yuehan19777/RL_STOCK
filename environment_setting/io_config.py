from __future__ import annotations
import os
import argparse
from typing import Any, Dict, Optional
import yaml


# 读取配置、解析路径、创建输出目录

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a dict.")
    return cfg

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    return ap.parse_args()

def resolve_path(path: str, project_root: str | None = None) -> str:
    if path is None:
        return path
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    base = project_root or os.getcwd()
    return os.path.abspath(os.path.join(base, path))

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
