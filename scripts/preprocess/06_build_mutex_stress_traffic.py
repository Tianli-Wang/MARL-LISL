#!/usr/bin/env python3
"""功能：兼容入口；构造更容易触发 future mutex 的 stress traffic pairs。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_mutex_stress_traffic import build_mutex_stress_traffic
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """读取 env.yaml 中的 traffic.stress 配置并保存 stress traffic 文件。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    args = parser.parse_args()
    build_mutex_stress_traffic(load_yaml(args.config), ROOT)


if __name__ == "__main__":
    main()
