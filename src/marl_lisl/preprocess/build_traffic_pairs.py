"""Generate fixed reachable source/destination traffic pairs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import networkx as nx
import numpy as np

from marl_lisl.store.graph_store import GraphStore
from marl_lisl.utils.progress import progress_iter


def _to_networkx(edge_index: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T)
    return graph


def _generate_one_set(
    graph: nx.Graph,
    num_flows: int,
    min_hops: int,
    max_retry: int,
    rng: np.random.Generator,
) -> np.ndarray:
    nodes = np.asarray(list(graph.nodes), dtype=np.int64)
    if len(nodes) < 2:
        raise RuntimeError("graph_0000 has fewer than two active nodes")
    pairs: list[tuple[int, int, float]] = []
    used: set[tuple[int, int]] = set()
    retries = 0
    while len(pairs) < num_flows and retries < max_retry:
        retries += 1
        source, dest = rng.choice(nodes, size=2, replace=False).tolist()
        key = (int(source), int(dest))
        if key in used:
            continue
        try:
            hops = nx.shortest_path_length(graph, source=source, target=dest)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if hops < min_hops:
            continue
        used.add(key)
        pairs.append((int(source), int(dest), 1.0))
    if len(pairs) != num_flows:
        raise RuntimeError(
            f"Could only generate {len(pairs)}/{num_flows} traffic pairs after {max_retry} retries"
        )
    return np.asarray(pairs, dtype=np.float64)


def _generate_named_set(args: tuple[str, nx.Graph, int, int, int, int]) -> tuple[str, np.ndarray]:
    name, graph, num_flows, min_hops, max_retry, seed = args
    rng = np.random.default_rng(seed)
    return name, _generate_one_set(graph, num_flows, min_hops, max_retry, rng)


def build_traffic_pairs(
    config: dict,
    project_root: Path,
    num_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    graph_store = GraphStore(project_root / config["graph_dir"], cache_size=1)
    graph = _to_networkx(graph_store.get_graph(0)["edge_index"])
    traffic_cfg = config["traffic"]
    base_seed = int(traffic_cfg["seed"])
    common = (
        graph,
        int(config["num_flows"]),
        int(traffic_cfg["min_source_dest_hops"]),
        int(traffic_cfg["max_retry"]),
    )
    tasks = [
        ("train", *common, base_seed),
        ("eval", *common, base_seed + 1),
    ]
    workers = max(1, int(num_workers or config.get("parallel_workers", 1)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            generated = dict(
                progress_iter(
                    executor.map(_generate_named_set, tasks),
                    total=len(tasks),
                    desc="04 生成 traffic pairs",
                    unit="set",
                )
            )
    else:
        generated = dict(
            progress_iter(
                (_generate_named_set(task) for task in tasks),
                total=len(tasks),
                desc="04 生成 traffic pairs",
                unit="set",
            )
        )
    train = generated["train"]
    eval_pairs = generated["eval"]

    traffic_dir = project_root / config["traffic_dir"]
    traffic_dir.mkdir(parents=True, exist_ok=True)
    np.save(project_root / config["traffic_train_path"], train)
    np.save(project_root / config["traffic_eval_path"], eval_pairs)
    metadata = {
        "seed": int(traffic_cfg["seed"]),
        "num_flows": int(config["num_flows"]),
        "demand": 1.0,
        "min_source_dest_hops": int(traffic_cfg["min_source_dest_hops"]),
        "train_num_sets": int(traffic_cfg["train_num_sets"]),
        "eval_num_sets": int(traffic_cfg["eval_num_sets"]),
    }
    with (traffic_dir / "traffic_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("train traffic pairs:\n", train)
    print("eval traffic pairs:\n", eval_pairs)
    return train, eval_pairs
