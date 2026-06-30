#!/usr/bin/env python3
"""功能：把 cand_XXXX.npz 候选路径打包为共享 memmap 查表文件。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.store import build_candidate_pack
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """Pack candidates for one or more traffic splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--split",
        choices=("train", "eval", "stress", "both", "all"),
        default="both",
    )
    parser.add_argument("--force", action="store_true", help="重新构建已有 pack")
    args = parser.parse_args()
    config = load_yaml(args.config)
    candidates_cfg = config["candidates"]
    if args.split == "both":
        splits = ("train", "eval")
    elif args.split == "all":
        splits = ("train", "eval", "stress")
    else:
        splits = (args.split,)
    for split in splits:
        key = {"train": "train_dir", "eval": "eval_dir", "stress": "stress_dir"}[split]
        build_candidate_pack(ROOT / candidates_cfg[key], force=args.force)


if __name__ == "__main__":
    main()
