from typing import Callable, Generator, Optional
from .graph import Graph, Node
import svgwrite
from svgwrite import px
from dataclasses import dataclass, field, replace

import numpy as np

@dataclass
class Production:

    left: Graph
    right: Graph
    seed_node: int
    attributes: Callable[[dict[int, Node]], dict[int, dict[str, str]]] = field(default = lambda _: {})
    predicate: Callable[[dict[int, Node]], bool] = field(default = lambda _: True)

    def inserted_ids(self) -> set[int]:
        return set(self.right.nodes.keys()) - set(self.left.nodes.keys())

    def modified_ids(self) -> set[int]:
        return set(self.right.nodes.keys()) & set(self.left.nodes.keys())

    def removed_ids(self) -> set[int]:
        return set(self.left.nodes.keys()) - set(self.right.nodes.keys())

    def is_applicable(self, graph: Graph, id_map: dict[int, int]) -> bool:
        attrs = {
            pattern_id: graph.node(graph_id)
            for pattern_id, graph_id in id_map.items()
            if pattern_id in self.left.nodes
        }
        return self.predicate(attrs)

    def apply(self, graph: Graph, id_map: dict[int, int]) -> Graph:
        if not self.is_applicable(graph, id_map):
            return graph

        A = np.zeros((6, 6))
        for i in range(3):
            A[i * 2, 0:2] = self.left.nodes[i][0].pos
            A[i * 2, 2] = 1
            A[i * 2 + 1, 3:5] = self.left.nodes[i][0].pos
            A[i * 2 + 1, 5] = 1

        coefs = np.linalg.solve(A, np.concatenate([graph.nodes[id_map[i]][0].pos for i in range(3)]))
        matrix = coefs.reshape((2, 3))

        def transform(p: np.ndarray) -> np.ndarray:
            return matrix @ np.concatenate((p, [1]))

        for i in range(3):
            assert(np.allclose(
                transform(self.left.nodes[i][0].pos),
                graph.nodes[id_map[i]][0].pos
            ))

        inserted_ids = self.inserted_ids()
        id_map = id_map.copy()
        free_id = graph.first_free_id()
        for i, id in enumerate(inserted_ids):
            id_map[id] = free_id + i

        new_nodes = graph.nodes.copy()

        for id in inserted_ids:
            new_id = id_map[id]
            node, ns = self.right.nodes[id]
            node = replace(node, pos = transform(node.pos))
            new_nodes[new_id] = (node, [id_map[n] for n in ns])

        for id in self.modified_ids():
            new_id = id_map[id]
            node, ns = self.right.nodes[id]
            node = replace(node, pos = transform(node.pos), attrs = graph.node(new_id).attrs)
            new_ns = set(id_map[n] for n in ns)
            new_ns = new_ns | set(graph.nodes[new_id][1])
            new_nodes[new_id] = (node, list(new_ns))

        for id in self.removed_ids():
            del new_nodes[id_map[id]]


        attrs = {
            pattern_id: graph.node(graph_id)
            for pattern_id, graph_id in id_map.items()
            if pattern_id in self.left.nodes
        }

        new_attrs = self.attributes(attrs)

        for id, attrs in new_attrs.items():
            new_nodes[id_map[id]][0].attrs = attrs

        return Graph(new_nodes)

    def apply_once(self, graph: Graph) -> Graph:
        subgraph = next(graph.find_isomorphic(self.left, self.seed_node))
        return self.apply(graph, subgraph)

    def apply_many(self, graph: Graph, max_applications: Optional[int] = None) -> Graph:
        i = 0
        while True:
            if max_applications is not None and i >= max_applications:
                break

            subgraph = next((g
                             for g in graph.find_isomorphic(self.left, self.seed_node)
                             if self.is_applicable(graph, g)),
                            None)
            if subgraph is None:
                break

            graph = self.apply(graph, subgraph)
            i += 1

        return graph

    def apply_all_possible(self, graph: Graph) -> Generator[Graph, None, None]:
        for subgraph in graph.find_isomorphic(self.left, self.seed_node):
            yield self.apply(graph, subgraph)

    def to_svg(self, dwg: svgwrite.Drawing) -> svgwrite.Drawing:
        left_svg = self.left.to_svg(dwg, show_ids = True)
        left_svg["x"] = -250 * px
        dwg.add(left_svg)
        right_svg = self.right.to_svg(dwg, show_ids = True)
        right_svg["x"] = 250 * px
        dwg.add(right_svg)

        dwg.add(dwg.line((420 * px, 200 * px), (480 * px, 200 * px), stroke = "black"))
        dwg.add(dwg.line((480 * px, 200 * px), (460 * px, 210 * px), stroke = "black"))
        dwg.add(dwg.line((480 * px, 200 * px), (460 * px, 190 * px), stroke = "black"))
        dwg.viewbox(0, 0, 900, 400)

        return dwg

    def _repr_svg_(self):
        dwg = svgwrite.Drawing(size = (900 * px, 400 * px))
        return self.to_svg(dwg).tostring()
