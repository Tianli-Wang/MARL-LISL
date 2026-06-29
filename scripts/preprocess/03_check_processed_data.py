#!/usr/bin/env python3
"""Check state arrays and randomly sampled graph snapshots."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.preprocess.check_processed_data import check_processed_data
from marl_lisl.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocess.yaml")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    check_processed_data(
        ROOT / cfg["sat_state_dir"],
        ROOT / cfg["graph_root_dir"] / cfg["d_max_name"],
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
