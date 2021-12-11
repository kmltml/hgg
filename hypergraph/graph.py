from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional
import numpy as np
import svgwrite
import svgwrite.container
from svgwrite import px

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

    def __init__(self, nodes: dict[int, tuple[Node, list[int]]]):
        self.nodes = nodes
        for a_id, (_, ns) in nodes.items():
            for b_id in ns:
                if a_id not in nodes[b_id][1]:
                    nodes[b_id][1].append(a_id)

    def node(self, id: int) -> Node:
        return self.nodes[id][0]

    def neighbours(self, id: int) -> list[int]:
        return self.nodes[id][1]

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

    def to_svg(self, dwg: svgwrite.Drawing, show_ids = False):
        svg = svgwrite.container.SVG()

        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0

        for node_id, (node, neighbours) in self.nodes.items():
            x, y = node.pos

            for n in neighbours:
                if n < node_id:
                    neighbour = self.node(n)
                    x2, y2 = neighbour.pos
                    svg.add(dwg.line((100 * x * px, 100 * y * px), (100 * x2 * px, 100 * y2 * px), stroke="black"))

        for node_id, (node, neighbours) in self.nodes.items():
            x, y = node.pos
            min_x = min(x * 100 - 40, min_x)
            min_y = min(y * 100 - 40, min_y)
            max_x = max(x * 100 + 40, max_x)
            max_y = max(y * 100 + 40, max_y)

            if node.is_hyperedge:
                size = 30
                pos = ((100 * x - size / 2) * px, (100 * y - size / 2) * px)
                svg.add(dwg.rect(insert = pos, size = (size * px, size * px), stroke = "black", fill = "white"))
                label = dwg.text(node.label, insert = (100 * x * px, 100 * y * px))
                label["text-anchor"] = "middle"
                label["dominant-baseline"] = "middle"
                svg.add(label)

                if show_ids:
                    id_label = dwg.text(str(node_id), insert = ((100 * x + size / 2 + 10) * px, (100 * y + size / 2 + 10) * px))
                    id_label["text-anchor"] = "middle"
                    id_label["dominant-baseline"] = "middle"
                    svg.add(id_label)
            else:
                svg.add(dwg.circle(center = (x * 100 * px, y * 100 * px), r = 2 * px))
                if node.label != "":
                    label = dwg.text(node.label, insert = ((100 * x - 10) * px, (100 * y - 10) * px))
                    label["text-anchor"] = "middle"
                    label["dominant-baseline"] = "middle"
                    svg.add(label)
                if show_ids:
                    id_label = dwg.text(str(node_id), insert = ((100 * x + 10) * px, (100 * y + 10) * px))
                    id_label["text-anchor"] = "middle"
                    id_label["dominant-baseline"] = "middle"
                    svg.add(id_label)

        svg.viewbox(min_x, min_y, max_x - min_x, max_y - min_y)
        return svg

    def _repr_svg_(self):
        dwg = svgwrite.Drawing(size = (400 * px, 400 * px))
        svg = self.to_svg(dwg)
        dwg.add(svg)
        return dwg.tostring()
