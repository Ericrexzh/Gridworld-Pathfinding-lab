"""BFS 独立手算预期、事件回放与回归测试。"""
import unittest
from collections import deque
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from grid import GridMap
from search import bfs, SearchEvent


class BFSTests(unittest.TestCase):
    def assert_path(self, m, result, steps):
        self.assertTrue(result.success)
        self.assertEqual(result.path_steps, steps)
        self.assertEqual(result.cost, steps * 10)
        self.assertEqual(result.path[0], m.start)
        self.assertEqual(result.path[-1], m.goal)
        self.assertEqual(len(set(result.path)), len(result.path))
        for a,b in zip(result.path,result.path[1:]):
            self.assertTrue(m.can_move(a,b,'4'))

    def test_direct(self):
        m = GridMap([[0,0]],(0,0),(0,1))
        r = bfs(m)
        self.assert_path(m,r,1)
        self.assertEqual(r.path,((0,0),(0,1)))

    def test_row(self):
        m = GridMap([[0]*5],(0,0),(0,4))
        self.assert_path(m,bfs(m),4)

    def test_open_square(self):
        m = GridMap([[0]*3 for _ in range(3)],(0,0),(2,2))
        self.assert_path(m,bfs(m),4)

    def test_order(self):
        m = GridMap([[0,0],[0,0]],(0,0),(1,1))
        r = bfs(m)
        self.assertEqual(r.path,((0,0),(0,1),(1,1)))
        self.assertEqual(r.events,(
            SearchEvent((0,0),((0,1),(1,0))),
            SearchEvent((0,1),((1,1),)),
            SearchEvent((1,0),()), SearchEvent((1,1),())))
        self.assertEqual((r.discovered_count,r.expanded_count),(4,4))

    def test_demo(self):
        m = GridMap.from_json(Path(__file__).resolve().parents[1]/'maps/demo.json')
        r = bfs(m)
        self.assert_path(m,r,8)
        self.assertEqual(r.path,((0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(3,4),(4,4)))

    def test_unreachable(self):
        r = bfs(GridMap([[0,1,0]],(0,0),(0,2)))
        self.assertFalse(r.success)
        self.assertEqual(r.path,())
        self.assertIsNone(r.cost)
        self.assertIsNone(r.path_steps)
        self.assertEqual(r.events,(SearchEvent((0,0),()),))
        self.assertEqual((r.expanded_count,r.discovered_count),(1,1))

    def test_detour(self):
        m = GridMap([[0,1,0],[0,1,0],[0,0,0]],(0,0),(0,2))
        r = bfs(m)
        self.assert_path(m,r,6)  # 下2、右2、上2，几何距离仅2
        self.assertEqual(r.path,((0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2)))

    def test_replay_and_deduplication(self):
        for goal in ((0,1),(2,2)):
            m = GridMap([[0]*3 for _ in range(3)],(0,0),goal)
            r = bfs(m)
            queue,seen,expanded = deque([m.start]),{m.start},set()
            for event in r.events:
                self.assertEqual(queue.popleft(),event.current)
                self.assertNotIn(event.current,expanded)
                expanded.add(event.current)
                for node in event.newly_discovered:
                    self.assertNotIn(node,seen)
                    self.assertTrue(m.can_move(event.current,node,'4'))
                    seen.add(node)
                    queue.append(node)
            self.assertEqual(r.discovered_count,1+sum(len(e.newly_discovered) for e in r.events))
            self.assertEqual(r.discovered_count,len(seen))
            self.assertEqual(r.expanded_count,len(r.events))
            self.assertLessEqual(r.expanded_count,r.discovered_count)
            self.assertTrue(set(r.path)<=seen)

    def test_goal_dequeue_no_successors(self):
        m = GridMap([[0,0],[0,0]],(0,0),(0,1))
        original = GridMap.neighbors
        calls = []
        def record(instance, position, mode):
            calls.append((position,mode))
            return original(instance,position,mode)
        with patch.object(GridMap,'neighbors',record):
            r = bfs(m)
        self.assertEqual(calls,[((0,0),'4')])
        self.assertEqual((r.discovered_count,r.expanded_count),(3,2))
        self.assertEqual(r.events[-1],SearchEvent(m.goal,()))

    def test_immutable_result(self):
        r = bfs(GridMap([[0,0]],(0,0),(0,1)))
        with self.assertRaises(FrozenInstanceError): r.cost = 0
        with self.assertRaises(FrozenInstanceError): r.events[0].current = (9,9)
        with self.assertRaises(TypeError): r.path[0] = (9,9)
        with self.assertRaises(TypeError): r.events[0].newly_discovered[0] = (9,9)

    def test_map_unchanged_repeatable(self):
        m = GridMap([[0,0],[0,0]],(0,0),(1,1))
        before = (m.grid,m.start,m.goal)
        self.assertEqual(bfs(m),bfs(m))
        self.assertEqual((m.grid,m.start,m.goal),before)

    def test_rectangles(self):
        for grid,goal,steps in [([[0],[0],[0]],(2,0),2),([[0]*4 for _ in range(2)],(1,3),4)]:
            m = GridMap(grid,(0,0),goal)
            self.assert_path(m,bfs(m),steps)

    def test_invalid_input(self):
        for value in (None,[],{},'map',1):
            with self.subTest(value=value),self.assertRaisesRegex(TypeError,'GridMap'):
                bfs(value)


if __name__ == '__main__':
    unittest.main()
