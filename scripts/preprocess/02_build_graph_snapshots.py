#!/usr/bin/env python3
"""Build sparse LISL graph snapshots from satellite states."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.build_graph_snapshots import build_graph_snapshots
from marl_lisl.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocess.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers; overrides parallel_workers in config")
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
