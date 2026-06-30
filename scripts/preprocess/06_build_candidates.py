#!/usr/bin/env python3
"""功能：离线预计算每个时隙、每条业务流的候选路径。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_candidates import build_candidates
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """读取环境配置，将 NetworkX 候选路径预先保存到 data/candidates。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--split",
        choices=("train", "eval", "stress", "both", "all"),
        default="both",
        help="预计算 train、eval、stress、train+eval 或全部候选路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行进程数；默认读取配置中的 parallel_workers",
    )
    args = parser.parse_args()
    build_candidates(load_yaml(args.config), ROOT, split=args.split, num_workers=args.workers)


if __name__ == "__main__":
    main()
