#!/usr/bin/env python3
"""功能：生成紧凑的一维节点互斥容量数组。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_mutex import build_node_mutex
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """读取 env.yaml 中的 future_mutex 配置并保存 node_mutex.npy。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    mutex_cfg = config["future_mutex"]
    output_path = ROOT / mutex_cfg["node_mutex_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capacity = build_node_mutex(
        int(config["num_sats"]), bool(mutex_cfg.get("strict_node_mutex", True))
    )
    np.save(output_path, capacity)
    metadata = {
        "num_sats": int(config["num_sats"]),
        "strict_node_mutex": bool(mutex_cfg.get("strict_node_mutex", True)),
        "default_node_capacity": int(capacity[0]),
        "shape": list(capacity.shape),
    }
    metadata_path = output_path.parent / "mutex_config.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved node mutex: {output_path}")
    print(f"shape={capacity.shape}, dtype={capacity.dtype}")
    print(f"saved config: {metadata_path}")


if __name__ == "__main__":
    main()
  
