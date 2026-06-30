"""Current-link outage checks; future mutex conflicts are intentionally absent."""

from marl_lisl.utils.graph import node_path_available


class ConflictDetector:
    def __init__(self, num_sats: int):
        self.num_sats = int(num_sats)

    def is_path_available(self, path: list[int] | None, graph: dict) -> bool:
        if path is None or len(path) < 2:
            return False
        return node_path_available(graph, path, self.num_sats)

    def count_outages(self, paths: list[list[int] | None], graph: dict) -> int:
        return sum(not self.is_path_available(path, graph) for path in paths)
