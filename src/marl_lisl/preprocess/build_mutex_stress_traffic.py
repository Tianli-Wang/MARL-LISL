"""Construct traffic pairs with shared relay nodes for future-mutex stress tests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

from marl_lisl.store.graph_store import GraphStore
from marl_lisl.utils.progress import progress_iter


def _to_networkx(edge_index: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T)
    return graph


def _sample_candidate_pairs(
    graph: nx.Graph,
    rng: np.random.Generator,
    num_candidate_pairs: int,
    max_paths_to_check: int,
) -> list[dict]:
    nodes = np.asarray(list(graph.nodes), dtype=np.int64)
    candidates: list[dict] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(num_candidate_pairs * 50, max_paths_to_check * 5)
    target = min(num_candidate_pairs, max_paths_to_check)
    iterator = progress_iter(range(max_attempts), desc="stress sample pairs", unit="try")
    for _ in iterator:
        if len(candidates) >= target:
            break
        attempts += 1
        if len(nodes) < 2:
            break
        source, dest = rng.choice(nodes, size=2, replace=False).tolist()
        key = (int(source), int(dest))
        if key in seen:
            continue
        seen.add(key)
        try:
            path = nx.shortest_path(graph, int(source), int(dest))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        relays = set(map(int, path[1:-1]))
        if not relays:
            continue
        candidates.append({
            "source": int(source),
            "dest": int(dest),
            "path": list(map(int, path)),
            "relays": relays,
        })
    if len(candidates) < target:
        warnings.warn(
            f"Only found {len(candidates)}/{target} candidate pairs after {attempts} attempts",
            stacklevel=2,
        )
    return candidates


def _select_shared_pairs(
    candidates: list[dict],
    num_flows: int,
    min_shared_relay_nodes: int,
) -> list[dict]:
    if not candidates:
        return []
    selected = [max(candidates, key=lambda item: len(item["relays"]))]
    remaining = [item for item in candidates if item is not selected[0]]
    relay_union = set(selected[0]["relays"])
    while remaining and len(selected) < num_flows:
        scored = []
        for item in remaining:
            overlap = len(item["relays"] & relay_union)
            scored.append((overlap, len(item["relays"]), item))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_overlap, _relay_count, best = scored[0]
        if best_overlap < min_shared_relay_nodes:
            warnings.warn(
                "Could not satisfy min_shared_relay_nodes for all flows; "
                "falling back to best available candidate. "
                "Consider increasing num_candidate_pairs or num_flows.",
                stacklevel=2,
            )
        selected.append(best)
        relay_union |= best["relays"]
        remaining.remove(best)
    if len(selected) < num_flows:
        warnings.warn(
            f"Only selected {len(selected)}/{num_flows} stress traffic pairs. "
            "Increase num_candidate_pairs or relax min_shared_relay_nodes.",
            stacklevel=2,
        )
    return selected


def build_mutex_stress_traffic(config: dict, project_root: Path) -> np.ndarray:
    """Build and save stress traffic pairs according to env.yaml."""
    project_root = Path(project_root)
    stress_cfg = config["traffic"]["stress"]
    graph_store = GraphStore(project_root / config["graph_dir"], cache_size=1, preload=False)
    target_step = int(stress_cfg.get("target_step", 0))
    graph = _to_networkx(graph_store.get_graph(target_step)["edge_index"])
    rng = np.random.default_rng(int(config["traffic"].get("seed", 42)))
    candidates = _sample_candidate_pairs(
        graph,
        rng,
        int(stress_cfg["num_candidate_pairs"]),
        int(stress_cfg["max_paths_to_check"]),
    )
    selected = _select_shared_pairs(
        candidates,
        int(stress_cfg["num_flows"]),
        int(stress_cfg["min_shared_relay_nodes"]),
    )
    demand = float(stress_cfg.get("demand", 1.0))
    pairs = np.asarray(
        [(item["source"], item["dest"], demand) for item in selected],
        dtype=np.float64,
    )
    output_path = project_root / stress_cfg.get("output_path", config["traffic_stress_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pairs)
    relay_counts: dict[int, int] = {}
    for item in selected:
        for relay in item["relays"]:
            relay_counts[relay] = relay_counts.get(relay, 0) + 1
    shared_nodes = {node: count for node, count in relay_counts.items() if count > 1}
    metadata = {
        "target_step": target_step,
        "num_flows_requested": int(stress_cfg["num_flows"]),
        "num_flows_saved": int(len(pairs)),
        "num_candidate_pairs_found": int(len(candidates)),
        "min_shared_relay_nodes": int(stress_cfg["min_shared_relay_nodes"]),
        "shared_relay_node_count": int(len(shared_nodes)),
        "max_relay_occupancy": int(max(shared_nodes.values(), default=0)),
        "output_path": str(output_path),
    }
    with (output_path.parent / "traffic_stress_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"saved stress traffic: {output_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if len(pairs) < int(stress_cfg["num_flows"]):
        print("WARNING: stress traffic has fewer flows than requested.")
    if not shared_nodes:
        print(
            "WARNING: No shared relay nodes found. Increase num_candidate_pairs, "
            "num_flows, or relax min_shared_relay_nodes."
        )
    return pairs
