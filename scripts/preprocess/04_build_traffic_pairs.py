#!/usr/bin/env python3
"""功能：从 graph_0000 中并行生成训练/评估可达业务源宿对。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_traffic_pairs import build_traffic_pairs
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """读取环境配置，生成 traffic_pairs_train/eval.npy 和元信息文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行线程数；覆盖配置文件中的 parallel_workers",
    )
    args = parser.parse_args()
    build_traffic_pairs(load_yaml(args.config), ROOT, num_workers=args.workers)


if __name__ == "__main__":
    main()
