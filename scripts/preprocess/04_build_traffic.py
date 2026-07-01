#!/usr/bin/env python3
"""功能：统一生成普通 train/eval traffic 与 future-mutex stress traffic。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_mutex_stress_traffic import build_mutex_stress_traffic
from marl_lisl.preprocess.build_traffic_pairs import build_traffic_pairs
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """按 split 生成普通源宿对、压力测试源宿对，或一次生成两者。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--split",
        choices=("normal", "stress", "all"),
        default="all",
        help="normal 生成 train/eval；stress 生成互斥压力流；all 依次生成全部",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="普通 train/eval traffic 生成线程数；stress 生成保持确定性串行逻辑",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.split in ("normal", "all"):
        build_traffic_pairs(config, ROOT, num_workers=args.workers)
    if args.split in ("stress", "all"):
        build_mutex_stress_traffic(config, ROOT)


if __name__ == "__main__":
    main()
