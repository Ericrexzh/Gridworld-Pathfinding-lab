"""Greedy：独立预期路径、纯 h 优先级和向后兼容。"""
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch
import search
from grid import GridMap
from search import bfs, dijkstra, greedy_best_first_search as greedy, SearchEvent


def nonoptimal_map():
    return GridMap([[0,0,0,0,1],[0,0,0,0,1],[0,1,0,1,0],[0,0,0,0,0]],(2,0),(2,4))


class GreedyTests(unittest.TestCase):
    def demo(self):
        return GridMap.from_json(Path(__file__).resolve().parents[1]/'maps/demo.json')

    def check_path(self,m,r):
        self.assertTrue(r.success)
        self.assertEqual((r.path[0],r.path[-1]),(m.start,m.goal))
        self.assertEqual(len(set(r.path)),len(r.path))
        self.assertEqual(r.path_steps,len(r.path)-1)
        self.assertEqual(r.cost,sum(m.move_cost(a,b,'4') for a,b in zip(r.path,r.path[1:])))

    def test_direct(self):
        r=greedy(GridMap([[0,0]],(0,0),(0,1)))
        self.assertEqual((r.path,r.path_steps,r.cost),(((0,0),(0,1)),1,10))

    def test_row(self):
        r=greedy(GridMap([[0]*5],(0,0),(0,4)))
        self.assertEqual((r.path_steps,r.cost),(4,40))

    def test_square(self):
        m=GridMap([[0,0],[0,0]],(0,0),(1,1)); r=greedy(m)
        self.check_path(m,r)
        self.assertEqual(r.path,((0,0),(0,1),(1,1)))
        self.assertEqual(r.events,(
            SearchEvent((0,0),((0,1),(1,0)),current_heuristic=20),
            SearchEvent((0,1),((1,1),),current_heuristic=10),
            SearchEvent((1,1),(),current_heuristic=0)))

    def test_demo(self):
        m=self.demo(); r=greedy(m)
        self.check_path(m,r)
        self.assertEqual((r.path_steps,r.cost,r.discovered_count,r.expanded_count),(8,80,12,9))
        self.assertEqual([e.current_heuristic for e in r.events],[80,70,60,50,40,30,20,10,0])

    def test_nonoptimal(self):
        m=nonoptimal_map(); r=greedy(m)
        expected=((2,0),(1,0),(1,1),(1,2),(2,2),(3,2),(3,3),(3,4),(2,4))
        self.assertEqual(r.path,expected)
        self.check_path(m,r)
        self.assertEqual((r.path_steps,r.cost),(8,80))
        for algorithm in (bfs,dijkstra):
            optimum=algorithm(m)
            self.assertEqual(optimum.path,((2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(2,4)))
            self.assertEqual((optimum.path_steps,optimum.cost),(6,60))
            self.assertGreater(r.cost,optimum.cost)

    def test_unreachable(self):
        r=greedy(GridMap([[0,1,0]],(0,0),(0,2)))
        self.assertFalse(r.success)
        self.assertEqual(r.path,())
        self.assertIsNone(r.cost)
        self.assertIsNone(r.path_steps)
        self.assertEqual(r.events,(SearchEvent((0,0),(),current_heuristic=20),))

    def test_heap_uses_only_h(self):
        m=nonoptimal_map(); pushed=[]; popped=[]
        push,pop=search.heapq.heappush,search.heapq.heappop
        def record_push(heap,item):
            pushed.append(item); return push(heap,item)
        def record_pop(heap):
            item=pop(heap); popped.append(item); return item
        with patch.object(search.heapq,'heappush',record_push),patch.object(search.heapq,'heappop',record_pop):
            r=greedy(m)
        self.assertEqual([item[1] for item in pushed],list(range(len(pushed))))
        self.assertEqual(len({item[2] for item in pushed}),len(pushed))
        for h,seq,pos in pushed:
            self.assertEqual(h,10*(abs(pos[0]-m.goal[0])+abs(pos[1]-m.goal[1])))
        self.assertEqual([(e.current_heuristic,e.current) for e in r.events],[(h,p) for h,s,p in popped])

    def test_tie_not_coordinate_order(self):
        # 下方 (2,1) 与左方 (1,0) 同 h=10；下先入堆但坐标字典序更大。
        m=GridMap([[0]*3 for _ in range(3)],(1,1),(2,0))
        r=greedy(m)
        self.assertEqual(r.path,((1,1),(2,1),(2,0)))
        self.assertEqual(tuple(e.current for e in r.events),r.path)

    def test_events_stats(self):
        for m in (self.demo(),nonoptimal_map(),GridMap([[0,1,0]],(0,0),(0,2))):
            r=greedy(m); seen={m.start}; expanded=set()
            for e in r.events:
                self.assertIn(e.current,seen)
                self.assertNotIn(e.current,expanded)
                expanded.add(e.current)
                self.assertEqual(e.current_heuristic,m.heuristic(e.current,m.goal,'4'))
                self.assertIsNone(e.current_cost)
                self.assertEqual(e.distance_updates,())
                for node in e.newly_discovered:
                    self.assertNotIn(node,seen)
                    self.assertTrue(m.can_move(e.current,node,'4'))
                    seen.add(node)
            self.assertEqual(r.discovered_count,len(seen))
            self.assertEqual(r.discovered_count,1+sum(len(e.newly_discovered) for e in r.events))
            self.assertEqual(r.expanded_count,len(r.events))
            self.assertLessEqual(r.expanded_count,r.discovered_count)
            if r.success:
                self.assertEqual(r.events[-1].current,m.goal)
                self.assertEqual(r.events[-1].newly_discovered,())

    def test_goal_not_expanded_further(self):
        m=GridMap([[0,0],[0,0]],(0,0),(0,1)); calls=[]
        original=GridMap.neighbors
        def record(instance,position,mode):
            calls.append((position,mode)); return original(instance,position,mode)
        with patch.object(GridMap,'neighbors',record): r=greedy(m)
        self.assertEqual(calls,[((0,0),'4')])
        self.assertEqual((r.discovered_count,r.expanded_count),(3,2))
        self.assertEqual(r.events[-1].current,m.goal)

    def test_repeat_and_map_unchanged(self):
        m=nonoptimal_map(); before=(m.grid,m.start,m.goal)
        self.assertEqual(greedy(m),greedy(m))
        self.assertEqual((m.grid,m.start,m.goal),before)

    def test_compatibility(self):
        self.assertIsNone(SearchEvent((0,0),()).current_heuristic)
        m=self.demo()
        for r in (bfs(m),dijkstra(m)):
            self.assertTrue(all(e.current_heuristic is None for e in r.events))
        self.assertTrue(all(e.current_cost is None and e.distance_updates==() for e in bfs(m).events))
        self.assertEqual(dijkstra(m).events[0].distance_updates[0].new_cost,10)

    def test_invalid(self):
        for value in (None,[],{},1,'map'):
            with self.subTest(value=value),self.assertRaisesRegex(TypeError,'GridMap'): greedy(value)

    def test_immutable(self):
        r=greedy(self.demo())
        with self.assertRaises(FrozenInstanceError): r.events[0].current_heuristic=0
        with self.assertRaises(TypeError): r.events[0].newly_discovered[0]=None


if __name__=='__main__': unittest.main()
