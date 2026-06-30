"""Current-link outage checks; future mutex conflicts are intentionally absent."""

from marl_lisl.utils.graph import build_edge_lookup, edge_path_from_node_path


class ConflictDetector:
    def __init__(self):
        self._graph_token: int | None = None
        self._edges: set[tuple[int, int]] = set()

    def _graph_edges(self, graph: dict) -> set[tuple[int, int]]:
        token = id(graph["edge_index"])
        if token != self._graph_token:
            self._edges = set(build_edge_lookup(graph))
            self._graph_token = token
        return self._edges

    def is_path_available(self, path: list[int] | None, graph: dict) -> bool:
        if path is None or len(path) < 2:
            return False
        return edge_path_from_node_path(path).issubset(self._graph_edges(graph))

    def count_outages(self, paths: list[list[int] | None], graph: dict) -> int:
        return sum(not self.is_path_available(path, graph) for path in paths)
