#!/usr/bin/env python3
"""功能：并行解析 STK 按时隙导出的卫星状态，生成固定编号状态数组。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_sat_state import build_sat_state
from marl_lisl.utils.io import load_yaml, resolve_raw_dir


def main() -> None:
    """读取预处理配置，调用底层并行转换函数生成 sat_state_m/valid_mask。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocess.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数；覆盖配置文件中的 parallel_workers")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    raw_dir = resolve_raw_dir(ROOT / cfg["raw_dir"], ROOT / cfg["fallback_raw_dir"])
    build_sat_state(
        raw_dir,
        ROOT / cfg["sat_state_dir"],
        int(cfg["expected_num_steps"]),
        int(cfg["expected_num_sats"]),
        num_workers=args.workers if args.workers is not None else int(cfg["parallel_workers"]),
    )


if __name__ == "__main__":
    main()
