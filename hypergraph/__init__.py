from .graph import *
from .production import *

def vertex(
        pos: tuple[int, int],
        label: str = "",
        neighbours: Optional[list[int]] = None,
        attrs: Optional[dict[str, str]] = None
) -> tuple[Node, list[int]]:
    attrs = attrs or {}
    neighbours = neighbours or []
    return Node.vertex(np.array(pos), label, attrs), neighbours

def hyperedge(
        pos: tuple[int, int],
        label: str,
        neighbours: list[int],
        attrs: Optional[dict[str, str]] = None,
) -> tuple[Node, list[int]]:
    attrs = attrs or {}
    return Node.hyperedge(np.array(pos), label, attrs), neighbours
