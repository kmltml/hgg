from __future__ import annotations

from dataclasses import dataclass
from typing import Generator
import numpy as np

@dataclass
class Node:

    is_hyperedge: bool
    pos: np.ndarray
    label: str
    attribs: dict[str, str]

    @staticmethod
    def vertex(pos: np.ndarray, label: str = "", attribs: dict[str, str] = {}) -> Node:
        return Node(False, pos, label, attribs)

    @staticmethod
    def hyperedge(pos: np.ndarray, label: str, attribs: dict[str, str] = {}) -> Node:
        return Node(True, pos, label, attribs)

    def matches(self, other: Node) -> bool:
        return self.is_hyperedge == other.is_hyperedge and self.label == other.label

@dataclass
class Graph:

    nodes: dict[int, tuple[Node, list[int]]]

    def node(self, id: int) -> Node:
        return self.nodes[id][0]

    def neighbours(self, id: int) -> list[int]:
        return self.nodes[id][1]

    def edge_neighbours(self, id: int) -> list[int]:
        return [x
                for e in self.neighbours(id)
                if self.node(e).is_hyperedge and self.node(e).label in {"s", "t", "r", "x", "e"}
                for x in self.neighbours(e)
                if x != id]

    def first_free_id(self) -> int:
        return max(self.nodes.keys()) + 1

    def find_isomorphic(self, pattern: Graph, seed_pattern_id: int) -> Generator[dict[int, int], None, None]:
        seed_pattern_node = pattern.node(seed_pattern_id)

        def rec(seed_id: int, map: dict[int, int], node_ids_to_add: list[int]) -> Generator[dict[int, int], None, None]:
            if len(node_ids_to_add) == 0:
                if all(id in map for id in pattern.nodes.keys()):
                    yield map
                    return

            added_id = node_ids_to_add.pop()
            added_pattern_node = pattern.node(added_id)
            required_neighbours = [map[id] for id in pattern.neighbours(added_id) if id in map]
            used_node_ids = set(map.values())

            for candidate_id in self.neighbours(required_neighbours[0]):
                if candidate_id in used_node_ids:
                    continue
                candidate_node = self.node(candidate_id)
                if (candidate_node.matches(added_pattern_node) and
                    all(nid in self.neighbours(candidate_id) for nid in required_neighbours)):

                    map[added_id] = candidate_id

                    next_node_ids_to_add = node_ids_to_add.copy()
                    for id in pattern.neighbours(added_id):
                        if id not in node_ids_to_add and id not in map:
                            next_node_ids_to_add.append(id)

                    for x in rec(seed_id, map | { added_id: candidate_id }, next_node_ids_to_add):
                        yield x

        for seed_id in (i for i, n in self.nodes.items() if n[0].matches(seed_pattern_node)):
            map = { seed_pattern_id: seed_id }
            node_ids_to_add = pattern.neighbours(seed_pattern_id).copy()

            for x in rec(seed_id, map, node_ids_to_add):
                yield x
