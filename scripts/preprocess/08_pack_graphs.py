#!/usr/bin/env python3
"""功能：把逐时隙 graph_XXXX.npz 打包为共享 memmap 图后端。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from marl_lisl.store import build_graph_pack
from marl_lisl.utils.config import load_yaml


def main() -> None:
    """Build the packed graph store used by high-throughput MAPPO training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/env.yaml")
    parser.add_argument("--force", action="store_true", help="重新构建已有 pack")
    args = parser.parse_args()
    config = load_yaml(args.config)
    graph_dir = ROOT / config["graph_dir"]
    pack_dir = ROOT / config.get("graph_pack_dir", graph_dir / "_packed")
    build_graph_pack(graph_dir, pack_dir=pack_dir, force=args.force)


if __name__ == "__main__":
    main()
