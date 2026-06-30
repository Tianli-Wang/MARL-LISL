"""K-shortest candidate path generation on sparse graph snapshots."""

from __future__ import annotations

from collections import Counter
from itertools import islice

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from marl_lisl.utils.graph import edge_ids_for_node_path


class PathGenerator:
    def __init__(self, num_candidates: int, weight_cfg: dict):
        self.num_candidates = int(num_candidates)
        self.weight_cfg = weight_cfg
        self._graph_token: int | None = None
        self._nx_graph: nx.Graph | None = None
        self._scipy_token: int | None = None
        self._scipy_inputs: dict | None = None

    def _edge_weights(self, edge_attr: np.ndarray) -> np.ndarray:
        cfg = self.weight_cfg
        return (
            float(cfg["propagation"]) * edge_attr[:, 1]
            + float(cfg["setup"]) * edge_attr[:, 2]
            + float(cfg["lifetime"])
            / (edge_attr[:, 3] + float(cfg["lifetime_epsilon"]))
        ).astype(np.float64, copy=False)

    def _get_search_graph(self, graph: dict[str, np.ndarray]) -> nx.Graph:
        token = id(graph["edge_index"])
        if token == self._graph_token and self._nx_graph is not None:
            return self._nx_graph
        edge_index, edge_attr = graph["edge_index"], graph["edge_attr"]
        weights = self._edge_weights(edge_attr)
        search_graph = nx.Graph()
        search_graph.add_weighted_edges_from(
            (int(u), int(v), float(weight))
            for (u, v), weight in zip(edge_index.T, weights)
        )
        self._graph_token = token
        self._nx_graph = search_graph
        return search_graph

    def _get_scipy_inputs(self, graph: dict[str, np.ndarray]) -> dict:
        token = id(graph["edge_index"])
        if token == self._scipy_token and self._scipy_inputs is not None:
            return self._scipy_inputs
        edge_index, edge_attr = graph["edge_index"], graph["edge_attr"]
        weights = self._edge_weights(edge_attr)
        num_nodes = int(
            self.weight_cfg.get(
                "num_sats",
                int(edge_index.max()) + 1 if edge_index.size else 0,
            )
        )
        rows = np.concatenate((edge_index[0], edge_index[1])).astype(np.int32, copy=False)
        cols = np.concatenate((edge_index[1], edge_index[0])).astype(np.int32, copy=False)
        base_data = np.concatenate((weights, weights)).astype(np.float64, copy=False)
        base_graph = csr_matrix((base_data, (rows, cols)), shape=(num_nodes, num_nodes))
        self._scipy_token = token
        self._scipy_inputs = {
            "graph": graph,
            "num_nodes": num_nodes,
            "edge_count": len(weights),
            "rows": rows,
            "cols": cols,
            "weights": weights,
            "base_data": base_data,
            "base_graph": base_graph,
        }
        return self._scipy_inputs

    def prepare_graph(self, graph: dict[str, np.ndarray]) -> nx.Graph:
        """Build or reuse the NetworkX search graph before parallel path queries."""
        if str(self.weight_cfg.get("engine", "scipy")).lower() == "networkx":
            return self._get_search_graph(graph)
        self._get_scipy_inputs(graph)
        return self._get_search_graph(graph) if self._nx_graph is not None else nx.Graph()

    @staticmethod
    def _edge_key(u: int, v: int) -> tuple[int, int]:
        u, v = int(u), int(v)
        return (u, v) if u < v else (v, u)

    def _generate_yen(self, search_graph: nx.Graph, source: int, dest: int) -> list[list[int]]:
        paths = nx.shortest_simple_paths(search_graph, source, dest, weight="weight")
        return [list(map(int, path)) for path in islice(paths, self.num_candidates)]

    @staticmethod
    def _path_from_predecessors(source: int, dest: int, predecessors: np.ndarray) -> list[int]:
        if dest >= len(predecessors) or source >= len(predecessors):
            return []
        path = [int(dest)]
        node = int(dest)
        while node != int(source):
            node = int(predecessors[node])
            if node < 0:
                return []
            path.append(node)
            if len(path) > len(predecessors):
                return []
        path.reverse()
        return path

    def _csr_with_penalty(self, inputs: dict, edge_use: Counter[int]) -> csr_matrix:
        if not edge_use:
            return inputs["base_graph"]
        weights = inputs["weights"].copy()
        penalty = float(self.weight_cfg.get("diversity_penalty", 1.0))
        edge_ids = np.fromiter(edge_use.keys(), dtype=np.int64, count=len(edge_use))
        counts = np.fromiter(edge_use.values(), dtype=np.float64, count=len(edge_use))
        weights[edge_ids] *= 1.0 + penalty * counts
        data = np.concatenate((weights, weights)).astype(np.float64, copy=False)
        return csr_matrix(
            (data, (inputs["rows"], inputs["cols"])),
            shape=(inputs["num_nodes"], inputs["num_nodes"]),
        )

    def _generate_scipy_penalized(
        self, graph: dict, source: int, dest: int
    ) -> list[list[int]]:
        inputs = self._get_scipy_inputs(graph)
        if source >= inputs["num_nodes"] or dest >= inputs["num_nodes"]:
            return []
        attempts = max(
            self.num_candidates,
            int(self.num_candidates * float(self.weight_cfg.get("max_attempts_multiplier", 4))),
        )
        edge_use: Counter[int] = Counter()
        seen: set[tuple[int, ...]] = set()
        results: list[list[int]] = []
        for _ in range(attempts):
            matrix = self._csr_with_penalty(inputs, edge_use)
            distances, predecessors = dijkstra(
                matrix,
                directed=False,
                indices=int(source),
                return_predecessors=True,
            )
            if not np.isfinite(distances[int(dest)]):
                break
            path = self._path_from_predecessors(int(source), int(dest), predecessors)
            if len(path) < 2:
                break
            edge_ids = edge_ids_for_node_path(
                graph,
                path,
                int(inputs["num_nodes"]),
            )
            if edge_ids is None:
                break
            edge_use.update(int(edge_id) for edge_id in edge_ids)
            path_key = tuple(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            results.append(path)
            if len(results) >= self.num_candidates:
                break
        return results

    def _generate_penalized(
        self, search_graph: nx.Graph, source: int, dest: int
    ) -> list[list[int]]:
        """Approximate K diverse paths with repeated Dijkstra and edge penalties."""
        penalty = float(self.weight_cfg.get("diversity_penalty", 1.0))
        attempts = max(
            self.num_candidates,
            int(self.num_candidates * float(self.weight_cfg.get("max_attempts_multiplier", 4))),
        )
        edge_use: Counter[tuple[int, int]] = Counter()
        seen: set[tuple[int, ...]] = set()
        results: list[list[int]] = []

        def weighted(u: int, v: int, data: dict) -> float:
            return float(data["weight"]) * (1.0 + penalty * edge_use[self._edge_key(u, v)])

        for _ in range(attempts):
            path = nx.shortest_path(search_graph, source, dest, weight=weighted)
            path_key = tuple(int(node) for node in path)
            for u, v in zip(path_key[:-1], path_key[1:]):
                edge_use[self._edge_key(u, v)] += 1
            if path_key in seen:
                continue
            seen.add(path_key)
            results.append(list(path_key))
            if len(results) >= self.num_candidates:
                break
        return results

    def generate(self, graph: dict, source: int, dest: int) -> list[list[int]]:
        if int(source) == int(dest):
            return []
        try:
            engine = str(self.weight_cfg.get("engine", "scipy")).lower()
            method = str(self.weight_cfg.get("method", "penalized_shortest")).lower()
            if engine != "networkx" and method not in {"yen", "shortest_simple_paths", "exact"}:
                paths = self._generate_scipy_penalized(graph, int(source), int(dest))
                return paths[: self.num_candidates]

            search_graph = self._get_search_graph(graph)
            if int(source) not in search_graph or int(dest) not in search_graph:
                return []
            if self.num_candidates == 1:
                path = nx.shortest_path(search_graph, int(source), int(dest), weight="weight")
                return [list(map(int, path))]
            if method in {"yen", "shortest_simple_paths", "exact"}:
                return self._generate_yen(search_graph, int(source), int(dest))
            return self._generate_penalized(search_graph, int(source), int(dest))
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError, ValueError):
            return []
