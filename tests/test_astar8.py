"""Round 7：手算预期、独立 Dijkstra、事件重放与真实网格数值回归。"""
import heapq
import math
import unittest
from itertools import count
from pathlib import Path
from unittest.mock import patch

from grid import GridMap
from search import a_star_search


def reference_dijkstra8(m, start=None):
    """仅供测试：不使用启发式或 A*，返回 (最短代价, 路径)。"""
    start = m.start if start is None else start
    distance, parent = {start: 0}, {start: None}
    serial = count()
    heap = [(0, next(serial), start)]
    while heap:
        cost, _, p = heapq.heappop(heap)
        if cost != distance[p]:
            continue
        if p == m.goal:
            path = []
            while p is not None:
                path.append(p)
                p = parent[p]
            return cost, tuple(reversed(path))
        for q, edge in m.neighbors(p, '8'):
            candidate = cost + edge
            if candidate < distance.get(q, math.inf):
                distance[q], parent[q] = candidate, p
                heapq.heappush(heap, (candidate, next(serial), q))
    return None, ()


def demo_map():
    return GridMap.from_json(Path(__file__).resolve().parents[1] / 'maps/demo.json')


def complex_map():
    # Round 6 的真实 7×7 障碍图；不修改邻接关系或边权。
    return GridMap([
        [0,0,0,0,0,0,0], [0,1,0,0,0,1,0], [1,0,0,0,0,0,0],
        [0,0,1,0,0,1,1], [0,0,0,0,1,0,0], [0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0]], (0,0), (6,6))


def floating_map():
    # Random(7) 探查中第 126 号图已固化；旧严格 < 出现伪改进。
    return GridMap([
        [0,1,0,1,0,1,0,1,1,0], [0,0,0,0,0,1,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0], [1,0,0,1,1,1,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0], [0,0,0,1,1,1,0,0,0,0],
        [1,0,0,0,0,0,0,0,1,0], [1,1,1,0,1,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0], [0,1,1,0,0,0,0,0,0,0]], (0,0), (9,9))


class AStar8Tests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9),
                        (actual, expected))

    def maps(self):
        return [demo_map(), complex_map(), floating_map(),
                GridMap([[0]*3 for _ in range(4)], (0,0), (3,2)),
                GridMap([[0,1],[1,0]], (0,0), (1,1))]

    def check_path(self, m, result, mode='8'):
        self.assertTrue(result.success)
        self.assertEqual((result.path[0], result.path[-1]), (m.start, m.goal))
        self.assertEqual(result.path_steps, len(result.path)-1)
        self.assertEqual(len(result.path), len(set(result.path)))
        cost = 0
        for a, b in zip(result.path, result.path[1:]):
            self.assertIn(b, dict(m.neighbors(a, mode)))
            cost += m.move_cost(a, b, mode)
            if a[0] != b[0] and a[1] != b[1]:
                self.assertTrue(m.is_walkable((a[0], b[1])))
                self.assertTrue(m.is_walkable((b[0], a[1])))
        self.close(result.cost, cost)

    def test_t01_default_compatibility(self):
        """默认与显式四方向的完整结果（包括事件）相同。"""
        for m in self.maps():
            self.assertEqual(a_star_search(m), a_star_search(m, mode='4'))

    def test_t02_open_square(self):
        m = GridMap([[0]*3 for _ in range(3)], (0,0), (2,2))
        r = a_star_search(m, '8')
        self.assertEqual(a_star_search(m).cost, 40)
        self.assertEqual(r.path, ((0,0),(1,1),(2,2)))
        self.assertEqual(r.path_steps, 2)
        self.close(r.cost, 28.28)
        self.close(r.events[0].current_heuristic, 28.28)
        self.check_path(m, r)

    def test_t03_asymmetric(self):
        m = GridMap([[0]*3 for _ in range(4)], (0,0), (3,2))
        r = a_star_search(m, '8')
        self.close(r.cost, 38.28)
        self.assertEqual(r.path_steps, 3)
        self.check_path(m, r)

    def test_t04_single_corner(self):
        m = GridMap([[0,1],[0,0]], (0,0), (1,1))
        self.assertNotIn(m.goal, dict(m.neighbors(m.start, '8')))
        r = a_star_search(m, '8')
        self.assertEqual(r.path, ((0,0),(1,0),(1,1)))
        self.close(r.cost, 20)
        self.check_path(m, r)

    def test_t05_double_corner(self):
        m = GridMap([[0,1],[1,0]], (0,0), (1,1))
        r = a_star_search(m, '8')
        self.assertEqual((r.success, r.path, r.cost, r.path_steps), (False,(),None,None))
        self.assertEqual((r.expanded_count, r.discovered_count), (1,1))
        self.assertEqual(r.events[0].current, m.start)
        self.assertEqual(r.events[0].distance_updates, ())

    def test_t06_blocked_diagonal_target(self):
        m = GridMap([[0,0,0],[0,1,0],[0,0,0]], (0,0), (2,2))
        r = a_star_search(m, '8')
        self.check_path(m, r)
        self.assertNotIn((1,1), dict(m.neighbors(m.start, '8')))
        for e in r.events:
            self.assertNotEqual(e.current, (1,1))
            self.assertNotIn((1,1), e.newly_discovered)
            self.assertNotIn((1,1), [u.position for u in e.distance_updates])

    def test_t07_demo(self):
        m = demo_map()
        self.assertEqual(a_star_search(m, '4').cost, 80)
        r = a_star_search(m, '8')
        self.close(r.cost, 74.14)
        self.close(reference_dijkstra8(m)[0], 74.14)
        self.check_path(m, r)

    def test_t08_modes(self):
        for mode in ('6', 4, 8, None, True, [], {}):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'mode'):
                a_star_search(demo_map(), mode)
        for mode in ('4', '8'):
            self.assertTrue(a_star_search(demo_map(), mode).success)

    def test_t09_path_cost(self):
        for m in self.maps()[:-1]:
            r = a_star_search(m, '8')
            self.check_path(m, r)
            cents = sum(1414 if a[0] != b[0] and a[1] != b[1] else 1000
                        for a, b in zip(r.path, r.path[1:]))
            self.close(r.cost, cents/100)

    def test_t10_heap_events_stale(self):
        """观察真实堆及邻居调用；逐事件重放 g，独立核对 h/f。"""
        m = complex_map()
        pushed, popped, calls = [], [], []
        push, pop, neighbors = heapq.heappush, heapq.heappop, GridMap.neighbors
        def record_push(heap, item):
            pushed.append(item)
            return push(heap, item)
        def record_pop(heap):
            expected = min(heap, key=lambda x: (x[0], x[1]))
            item = pop(heap)
            self.assertEqual(item, expected)
            popped.append(item)
            return item
        def query(instance, p, mode):
            self.assertEqual(mode, '8')
            calls.append(p)
            return neighbors(instance, p, mode)
        with patch.object(heapq, 'heappush', record_push), \
             patch.object(heapq, 'heappop', record_pop), \
             patch.object(GridMap, 'neighbors', query):
            r = a_star_search(m, '8')
        self.assertEqual([x[1] for x in pushed], list(range(len(pushed))))
        for f, _, g, p in pushed:
            dr, dc = abs(p[0]-m.goal[0]), abs(p[1]-m.goal[1])
            self.close(f, g+14.14*min(dr,dc)+10*abs(dr-dc))
        dist, settled, stale, improvements = {m.start: 0}, set(), [], []
        events = iter(r.events)
        for f, _, g, p in popped:
            if g != dist[p]:
                stale.append((p, g, dist[p]))
                continue
            self.assertNotIn(p, settled)
            settled.add(p)
            e = next(events)
            self.assertEqual((e.current, e.current_cost), (p, g))
            self.close(e.current_heuristic, m.heuristic(p, m.goal, '8'))
            self.close(f, e.current_cost+e.current_heuristic)
            first = []
            for u in e.distance_updates:
                self.assertEqual(u.previous_cost, dist.get(u.position))
                self.close(u.new_cost, g+m.move_cost(p, u.position, '8'))
                if u.previous_cost is None:
                    first.append(u.position)
                else:
                    self.assertGreater(u.previous_cost-u.new_cost, 1e-9)
                    improvements.append(u)
                dist[u.position] = u.new_cost
            self.assertEqual(e.newly_discovered, tuple(first))
        self.assertIsNone(next(events, None))
        self.assertGreater(len(stale), 0)  # 必须真的弹出旧项，不能只验证生成。
        self.assertGreater(len(improvements), 0)
        self.assertEqual(calls, [e.current for e in r.events[:-1]])
        self.assertEqual(len(popped), r.expanded_count+len(stale))
        self.assertEqual((r.discovered_count,r.expanded_count), (len(dist),len(settled)))
        self.assertEqual(r.discovered_count, 1+sum(len(e.newly_discovered) for e in r.events))
        self.assertEqual(r.events[-1].current, m.goal)
        self.assertEqual(r.events[-1].distance_updates, ())
        self.assertEqual(r.events[-1].newly_discovered, ())
        self.close(r.events[-1].current_heuristic, 0)
        self.close(r.cost, reference_dijkstra8(m)[0])

    def test_t11_reference(self):
        for m in self.maps():
            expected, path = reference_dijkstra8(m)
            r = a_star_search(m, '8')
            self.assertEqual(r.success, expected is not None)
            if r.success:
                self.close(r.cost, expected)
                self.check_path(m, r)
                self.close(expected, sum(m.move_cost(a,b,'8') for a,b in zip(path,path[1:])))

    def test_t12_exhaustive_128(self):
        cells = [(r,c) for r in range(3) for c in range(3) if (r,c) not in ((0,0),(2,2))]
        for mask in range(128):
            grid = [[0]*3 for _ in range(3)]
            for i, (r,c) in enumerate(cells):
                grid[r][c] = (mask >> i) & 1
            m = GridMap(grid, (0,0), (2,2))
            with self.subTest(mask=mask):
                a, b = a_star_search(m, '4'), a_star_search(m, '8')
                expected, _ = reference_dijkstra8(m)
                self.assertEqual(a.success, b.success)
                self.assertEqual(b.success, expected is not None)
                if b.success:
                    self.close(b.cost, expected)
                    self.assertLessEqual(b.cost, a.cost+1e-9)
                    self.check_path(m, b)
                else:
                    self.assertEqual((b.path,b.cost,b.path_steps), ((),None,None))

    def test_t13_heuristic(self):
        for m in self.maps():
            self.close(m.heuristic(m.goal,m.goal,'8'), 0)
            for row in range(m.height):
                for col in range(m.width):
                    p = row, col
                    if not m.is_walkable(p):
                        continue
                    h = m.heuristic(p,m.goal,'8')
                    for q, edge in m.neighbors(p,'8'):
                        self.assertLessEqual(h, edge+m.heuristic(q,m.goal,'8')+1e-9)
                    cost, _ = reference_dijkstra8(m, start=p)
                    self.assertLessEqual(h, math.inf if cost is None else cost+1e-9)

    def test_t14_float_stability(self):
        """固化伪改进反例，另测长对角线与多条混合等价路线。"""
        maps = [floating_map(), GridMap([[0]*30 for _ in range(30)], (0,0),(29,29)),
                GridMap([[0]*12 for _ in range(20)], (0,0),(19,11))]
        for m in maps:
            r = a_star_search(m, '8')
            self.close(r.cost, reference_dijkstra8(m)[0])
            self.check_path(m, r)
            exact = {m.start: 0}
            self.assertEqual(len({e.current for e in r.events}), len(r.events))
            for e in r.events:
                for u in e.distance_updates:
                    p, q = e.current, u.position
                    cents = exact[p] + (1414 if p[0]!=q[0] and p[1]!=q[1] else 1000)
                    if q in exact:
                        self.assertLess(cents, exact[q])  # 整数核对，拒绝理论等价伪更新。
                    exact[q] = cents
                    self.close(u.new_cost, cents/100)
        updates = [u for e in a_star_search(floating_map(),'8').events
                   for u in e.distance_updates if u.position == (9,7)]
        self.assertEqual(len(updates), 1)
        self.close(updates[0].new_cost, 148.28)

    def test_t15_tie_and_repeat(self):
        # 终点隔绝，使可达区域的候选全部出队，观察真实同 f 平局。
        m = GridMap([[0,0,0],[0,0,0],[0,0,0],[1,1,1],[0,0,0]], (1,1),(4,0))
        pushed = []
        push = heapq.heappush
        def record(heap, item):
            pushed.append(item)
            return push(heap,item)
        with patch.object(heapq,'heappush',record):
            r = a_star_search(m,'8')
        self.assertEqual(r, a_star_search(m,'8'))
        # 找到真实同 f、先入堆坐标却更大的节点对，证明不是坐标破平局。
        order = [e.current for e in r.events]
        pairs = [(a,b) for i,a in enumerate(pushed) for b in pushed[i+1:]
                 if a[0]==b[0] and a[3]>b[3] and a[3]!=b[3]]
        self.assertTrue(pairs)
        self.assertTrue(any(order.index(a[3]) < order.index(b[3]) for a,b in pairs))
        for m in self.maps():
            self.assertEqual(a_star_search(m,'8'), a_star_search(m,'8'))

    def test_t16_failure_stats(self):
        m = GridMap([[0,0,1,0],[0,0,1,0]], (0,0),(0,3))
        r = a_star_search(m,'8')
        self.assertEqual((r.success,r.path,r.cost,r.path_steps), (False,(),None,None))
        self.assertEqual((r.expanded_count,r.discovered_count,len(r.events)), (4,4,4))
        self.assertEqual({e.current for e in r.events}, {(0,0),(0,1),(1,0),(1,1)})

    def test_t17_immutable(self):
        for m in self.maps():
            before = m.grid, m.start, m.goal
            for mode in ('4','8'):
                a_star_search(m,mode)
            self.assertEqual((m.grid,m.start,m.goal), before)


if __name__ == '__main__':
    unittest.main()
