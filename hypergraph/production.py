from .graph import Graph
from dataclasses import dataclass, replace

import numpy as np

@dataclass
class Production:

    left: Graph
    right: Graph
    seed_node: int

    def inserted_ids(self) -> set[int]:
        return set(self.right.nodes.keys()) - set(self.left.nodes.keys())

    def modified_ids(self) -> set[int]:
        return set(self.right.nodes.keys()) & set(self.left.nodes.keys())

    def removed_ids(self) -> set[int]:
        return set(self.left.nodes.keys()) - set(self.right.nodes.keys())

    def apply(self, graph: Graph, id_map: dict[int, int]) -> Graph:
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
            node = replace(node, pos = transform(node.pos))
            new_ns = set(id_map[n] for n in ns)
            new_ns = new_ns | set(graph.nodes[new_id][1])
            new_nodes[new_id] = (node, list(new_ns))

        for id in self.removed_ids():
            del new_nodes[id_map[id]]

        return Graph(new_nodes)
