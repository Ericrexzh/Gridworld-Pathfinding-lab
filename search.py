"""四方向BFS、Dijkstra、Greedy及四/八方向 A*,共享搜索结果"""
import heapq
from math import isclose
from itertools import count
from collections import deque
from dataclasses import dataclass

from grid import GridMap, ORTHOGONAL_COST

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DistanceUpdate:
    """一次严格距离松弛,首次发现previous_cost为None"""
    position: Position
    previous_cost: int | float | None
    new_cost: int | float


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """一次正式出队,产生的所有坐标和序列均为元组"""
    current: Position
    newly_discovered: tuple[Position, ...]
    current_cost: int | float | None = None
    distance_updates: tuple[DistanceUpdate, ...] = ()
    current_heuristic: int | float | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    success: bool
    path: tuple[Position, ...]
    cost: int | float | None
    events: tuple[SearchEvent, ...]
    discovered_count: int

    @property
    def expanded_count(self) -> int:
        return len(self.events)

    @property
    def path_steps(self) -> int | None:
        return len(self.path) - 1 if self.success else None


def _reconstruct_path(parent: dict, goal: Position) -> tuple[Position, ...]:
    """沿搜索构建的无环前驱链回溯,仅在成功时调用"""
    reverse_path = []
    current = goal
    while current is not None:
        reverse_path.append(current)
        current = parent[current]
    return tuple(reversed(reverse_path))


def bfs(grid_map: GridMap) -> SearchResult:
    """按四方向最少边数搜索,终点出队时停止,失败仍保留事件"""
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    queue = deque([grid_map.start])
    visited = {grid_map.start}
    parent = {grid_map.start: None}
    events = []
    while queue:
        current = queue.popleft()
        if current == grid_map.goal:
            events.append(SearchEvent(current, ()))
            path = _reconstruct_path(parent, current)
            return SearchResult(True, path, ORTHOGONAL_COST * (len(path) - 1),
                                tuple(events), len(visited))
        newly_discovered = []
        for next_node, _cost in grid_map.neighbors(current, mode="4"):
            if next_node not in visited:
                visited.add(next_node)
                parent[next_node] = current
                queue.append(next_node)
                newly_discovered.append(next_node)
        events.append(SearchEvent(current, tuple(newly_discovered)))
    return SearchResult(False, (), None, tuple(events), len(visited))


def dijkstra(grid_map: GridMap) -> SearchResult:
    """非负边权下按累计代价扩展,接口固定使用四方向"""
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    sequence = count()
    heap = []
    heapq.heappush(heap, (0, next(sequence), grid_map.start))
    dist = {grid_map.start: 0}
    parent = {grid_map.start: None}
    settled = set()
    events = []
    while heap:
        current_cost, _sequence, current = heapq.heappop(heap)
        if current_cost != dist[current]:
            continue
        if current in settled:
            continue
        settled.add(current)
        if current == grid_map.goal:
            events.append(SearchEvent(current, (), current_cost))
            return SearchResult(True, _reconstruct_path(parent, current), current_cost,
                                tuple(events), len(dist))
        newly_discovered = []
        distance_updates = []
        for neighbor, edge_cost in grid_map.neighbors(current, mode="4"):
            if neighbor in settled:
                continue
            new_cost = dist[current] + edge_cost
            previous_cost = dist.get(neighbor)
            if previous_cost is None or new_cost < previous_cost:
                if previous_cost is None:
                    newly_discovered.append(neighbor)
                dist[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(heap, (new_cost, next(sequence), neighbor))
                distance_updates.append(DistanceUpdate(neighbor, previous_cost, new_cost))
        events.append(SearchEvent(current, tuple(newly_discovered), current_cost,
                                  tuple(distance_updates)))
    return SearchResult(False, (), None, tuple(events), len(dist))


def greedy_best_first_search(grid_map: GridMap) -> SearchResult:
    """仅按h排序,首次发现即固定前驱,返回路径不保证最优"""
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    sequence = count()
    heap = []
    start_h = grid_map.heuristic(grid_map.start, grid_map.goal, mode="4")
    heapq.heappush(heap, (start_h, next(sequence), grid_map.start))
    visited = {grid_map.start}
    parent = {grid_map.start: None}
    events = []
    while heap:
        current_h, _sequence, current = heapq.heappop(heap)
        if current == grid_map.goal:
            events.append(SearchEvent(current, (), current_heuristic=current_h))
            path = _reconstruct_path(parent, current)
            cost = sum(grid_map.move_cost(a, b, mode="4")
                       for a, b in zip(path, path[1:]))
            return SearchResult(True, path, cost, tuple(events), len(visited))
        newly_discovered = []
        for neighbor, _edge_cost in grid_map.neighbors(current, mode="4"):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            h = grid_map.heuristic(neighbor, grid_map.goal, mode="4")
            heapq.heappush(heap, (h, next(sequence), neighbor))
            newly_discovered.append(neighbor)
        events.append(SearchEvent(current, tuple(newly_discovered),
                                  current_heuristic=current_h))
    return SearchResult(False, (), None, tuple(events), len(visited))



def a_star_search(grid_map: GridMap, mode: str = "4") -> SearchResult:
    """四/八方向A*,依赖匹配的一致启发式"""
    if not isinstance(grid_map, GridMap):
        raise TypeError("grid_map must be a GridMap")
    sequence = count()
    heap = []
    start_h = grid_map.heuristic(grid_map.start, grid_map.goal, mode=mode)
    heapq.heappush(heap, (start_h, next(sequence), 0, grid_map.start))
    dist = {grid_map.start: 0}
    parent = {grid_map.start: None}
    settled = set()
    events = []
    while heap:
        _current_f, _sequence, current_g, current = heapq.heappop(heap)
        if current_g != dist[current]:
            continue
        if current in settled:
            continue
        settled.add(current)
        current_h = grid_map.heuristic(current, grid_map.goal, mode=mode)
        if current == grid_map.goal:
            events.append(SearchEvent(current, (), current_g,
                                      current_heuristic=current_h))
            return SearchResult(True, _reconstruct_path(parent, current), current_g,
                                tuple(events), len(dist))
        newly_discovered = []
        distance_updates = []
        for neighbor, edge_cost in grid_map.neighbors(current, mode=mode):
            if neighbor in settled:
                continue
            new_g = current_g + edge_cost
            old_g = dist.get(neighbor)
            #八方向等价路线的加法顺序可能相差几个浮点末位
            if old_g is None or (new_g < old_g and not (
                    mode == "8" and isclose(new_g, old_g, rel_tol=0.0, abs_tol=1e-9))):
                if old_g is None:
                    newly_discovered.append(neighbor)
                dist[neighbor] = new_g
                parent[neighbor] = current
                h = grid_map.heuristic(neighbor, grid_map.goal, mode=mode)
                heapq.heappush(heap, (new_g + h, next(sequence), new_g, neighbor))
                distance_updates.append(DistanceUpdate(neighbor, old_g, new_g))
        events.append(SearchEvent(current, tuple(newly_discovered), current_g,
                                  tuple(distance_updates), current_h))
    return SearchResult(False, (), None, tuple(events), len(dist))
