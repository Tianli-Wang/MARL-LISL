#!/usr/bin/env python3
"""功能：并行构建逐时隙 LISL 稀疏动态图快照。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_graph_snapshots import build_graph_snapshots
from marl_lisl.utils.io import load_yaml


def main() -> None:
    """读取卫星状态数组，并按配置中的 d_max 等参数生成 graph_XXXX.npz。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocess.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数；覆盖配置文件中的 parallel_workers")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    build_graph_snapshots(
        ROOT / cfg["sat_state_dir"],
        ROOT / cfg["graph_root_dir"] / cfg["d_max_name"],
        d_max_m=float(cfg["d_max_m"]),
        earth_radius_m=float(cfg["earth_radius_m"]),
        speed_of_light_m_s=float(cfg["speed_of_light_m_s"]),
        dt=float(cfg["dt"]),
        default_capacity=float(cfg["default_capacity"]),
        setup_delay_cfg=cfg["setup_delay"],
        expected_num_steps=int(cfg["expected_num_steps"]),
        expected_num_sats=int(cfg["expected_num_sats"]),
        num_workers=args.workers if args.workers is not None else int(cfg["parallel_workers"]),
    )


if __name__ == "__main__":
    main()
