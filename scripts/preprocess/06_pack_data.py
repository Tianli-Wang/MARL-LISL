#!/usr/bin/env python3
"""功能：统一把图快照和离线候选路径打包成共享 memmap 查找表。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.store import build_candidate_pack, build_graph_pack
from marl_lisl.utils.config import load_yaml


def _selected_splits(value: str) -> tuple[str, ...]:
    """把便捷 split 名称展开为需要逐个处理的实际数据集合。"""
    if value == "both":
        return "train", "eval"
    if value == "all":
        return "train", "eval", "stress"
    return (value,)


def main() -> None:
    """按 target 打包图、候选路径或二者，并允许强制覆盖已有 pack。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument(
        "--target", choices=("graphs", "candidates", "all"), default="all"
    )
    parser.add_argument(
        "--split", choices=("train", "eval", "stress", "both", "all"), default="both"
    )
    parser.add_argument("--force", action="store_true", help="重新构建已有 pack")
    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.target in ("graphs", "all"):
        graph_dir = ROOT / config["graph_dir"]
        pack_dir = ROOT / config.get("graph_pack_dir", graph_dir / "_packed")
        build_graph_pack(graph_dir, pack_dir=pack_dir, force=args.force)

    if args.target in ("candidates", "all"):
        candidates_cfg = config["candidates"]
        candidate_keys = {
            "train": "train_dir",
            "eval": "eval_dir",
            "stress": "stress_dir",
        }
        for split in _selected_splits(args.split):
            # 每套 traffic 必须读取与自身源宿对一致的候选目录，不能把 train
            # candidates 打包后误用于 eval/stress 环境。
            build_candidate_pack(
                ROOT / candidates_cfg[candidate_keys[split]], force=args.force
            )


if __name__ == "__main__":
    main()
