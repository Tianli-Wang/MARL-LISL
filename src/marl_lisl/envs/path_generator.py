"""K-shortest candidate path generation on sparse graph snapshots."""

from __future__ import annotations

from itertools import islice

import networkx as nx
import numpy as np


class PathGenerator:
    def __init__(self, num_candidates: int, weight_cfg: dict):
        self.num_candidates = int(num_candidates)
        self.weight_cfg = weight_cfg
        self._graph_token: int | None = None
        self._nx_graph: nx.Graph | None = None

    def _get_search_graph(self, graph: dict[str, np.ndarray]) -> nx.Graph:
        token = id(graph["edge_index"])
        if token == self._graph_token and self._nx_graph is not None:
            return self._nx_graph
        edge_index, edge_attr = graph["edge_index"], graph["edge_attr"]
        cfg = self.weight_cfg
        weights = (
            float(cfg["propagation"]) * edge_attr[:, 1]
            + float(cfg["setup"]) * edge_attr[:, 2]
            + float(cfg["lifetime"])
            / (edge_attr[:, 3] + float(cfg["lifetime_epsilon"]))
        )
        search_graph = nx.Graph()
        search_graph.add_weighted_edges_from(
            (int(u), int(v), float(weight))
            for (u, v), weight in zip(edge_index.T, weights)
        )
        self._graph_token = token
        self._nx_graph = search_graph
        return search_graph

    def prepare_graph(self, graph: dict[str, np.ndarray]) -> nx.Graph:
        """Build or reuse the NetworkX search graph before parallel path queries."""
        return self._get_search_graph(graph)

    def generate(self, graph: dict, source: int, dest: int) -> list[list[int]]:
        if int(source) == int(dest):
            return []
        try:
            search_graph = self._get_search_graph(graph)
            if source not in search_graph or dest not in search_graph:
                return []
            if self.num_candidates == 1:
                path = nx.shortest_path(
                    search_graph, int(source), int(dest), weight="weight"
                )
                return [list(map(int, path))]
            paths = nx.shortest_simple_paths(search_graph, int(source), int(dest), weight="weight")
            return [list(map(int, path)) for path in islice(paths, self.num_candidates)]
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError, ValueError):
            return []
