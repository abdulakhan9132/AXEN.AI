from __future__ import annotations

from collections import deque


def transitive_reachability(edges: list[tuple[int, int]], start: int) -> set[int]:
    graph: dict[int, list[int]] = {}
    for source, target in edges: graph.setdefault(source, []).append(target)
    reached, queue = set(), deque([start])
    while queue:
        node = queue.popleft()
        if node in reached: continue
        reached.add(node); queue.extend(graph.get(node, []))
    return reached


def shortest_path(edges: list[tuple[int, int]], start: int, goal: int) -> int | None:
    graph: dict[int, list[int]] = {}
    for source, target in edges: graph.setdefault(source, []).append(target)
    queue = deque([(start, 0)]); visited = {start}
    while queue:
        node, distance = queue.popleft()
        if node == goal: return distance
        for neighbor in graph.get(node, []):
            if neighbor not in visited: visited.add(neighbor); queue.append((neighbor, distance + 1))
    return None
