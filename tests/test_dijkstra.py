"""真实等权地图与临时模拟非负权图分别验证。"""
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch
import search
from grid import GridMap
from search import bfs, dijkstra, SearchEvent, DistanceUpdate

S,A,B,C,D,G = (0,0),(0,1),(1,0),(1,1),(0,2),(1,2)
WEIGHTED_GRAPH = {
    S: [(A,10),(B,1)], A: [(S,10),(C,1),(D,20)],
    B: [(S,1),(C,1)], C: [(B,1),(A,1),(G,30)],
    D: [(A,20),(G,1)], G: [(C,30),(D,1)],
}


class DijkstraTests(unittest.TestCase):
    def demo(self):
        return GridMap.from_json(Path(__file__).resolve().parents[1]/'maps/demo.json')

    def check_path(self,m,steps):
        r=dijkstra(m)
        self.assertTrue(r.success)
        self.assertEqual((r.path_steps,r.cost),(steps,steps*10))
        self.assertEqual((r.path[0],r.path[-1]),(m.start,m.goal))
        self.assertEqual(len(r.path),len(set(r.path)))
        self.assertEqual(r.cost,sum(m.move_cost(a,b,'4') for a,b in zip(r.path,r.path[1:])))
        return r

    def test_direct(self): self.check_path(GridMap([[0,0]],S,A),1)
    def test_row(self): self.check_path(GridMap([[0]*5],S,(0,4)),4)
    def test_square(self):
        r=self.check_path(GridMap([[0,0],[0,0]],S,C),2)
        self.assertEqual(r.path,(S,A,C))
        self.assertEqual(tuple(e.current for e in r.events),(S,A,B,C))

    def test_demo(self):
        m=self.demo()
        self.assertEqual(self.check_path(m,8).path,bfs(m).path)

    def test_detour(self):
        self.check_path(GridMap([[0,1,0],[0,1,0],[0,0,0]],S,(0,2)),6)

    def test_failure(self):
        r=dijkstra(GridMap([[0,1,0]],S,(0,2)))
        self.assertFalse(r.success)
        self.assertEqual(r.path,())
        self.assertIsNone(r.cost)
        self.assertIsNone(r.path_steps)
        self.assertEqual(r.events,(SearchEvent(S,(),0),))

    def test_event_replay_and_stats(self):
        m=self.demo(); r=dijkstra(m)
        dist={m.start:0}; expanded=set()
        for e in r.events:
            self.assertEqual(e.current_cost,dist[e.current])
            self.assertNotIn(e.current,expanded)
            expanded.add(e.current)
            first=[]
            for u in e.distance_updates:
                self.assertEqual(u.previous_cost,dist.get(u.position))
                self.assertEqual(u.new_cost,e.current_cost+m.move_cost(e.current,u.position,'4'))
                if u.previous_cost is None: first.append(u.position)
                else: self.assertLess(u.new_cost,u.previous_cost)
                dist[u.position]=u.new_cost
            self.assertEqual(tuple(first),e.newly_discovered)
        self.assertEqual(r.discovered_count,len(dist))
        self.assertEqual(r.discovered_count,1+sum(len(e.newly_discovered) for e in r.events))
        self.assertEqual(r.expanded_count,len(expanded))
        self.assertLessEqual(r.expanded_count,r.discovered_count)

    def test_equal_cost_no_update(self):
        r=dijkstra(GridMap([[0,0],[0,0]],S,C))
        self.assertEqual(r.events[0],SearchEvent(S,(A,B),0,
                         (DistanceUpdate(A,None,10),DistanceUpdate(B,None,10))))
        self.assertEqual(r.events[2],SearchEvent(B,(),10,()))
        self.assertEqual(r.events[-1],SearchEvent(C,(),20,()))

    def test_monotonic(self):
        costs=[e.current_cost for e in dijkstra(self.demo()).events]
        self.assertEqual(costs,sorted(costs))

    def test_weighted_stale(self):
        m=GridMap([[0]*3 for _ in range(2)],S,G)
        calls=[]; popped=[]; original=search.heapq.heappop
        def neighbors(instance,position,mode):
            self.assertEqual(mode,'4'); calls.append(position)
            return WEIGHTED_GRAPH[position]
        def pop(heap):
            item=original(heap); popped.append((item[2],item[0])); return item
        with patch.object(GridMap,'neighbors',neighbors),patch.object(search.heapq,'heappop',pop):
            r=dijkstra(m)
        self.assertEqual(r.path,(S,B,C,A,D,G))
        self.assertEqual(r.cost,24)
        self.assertEqual(popped,[(S,0),(B,1),(C,2),(A,3),(A,10),(D,23),(G,24)])
        self.assertEqual(calls,[S,B,C,A,D])
        self.assertEqual(tuple(e.current_cost for e in r.events),(0,1,2,3,23,24))
        self.assertEqual((r.expanded_count,r.discovered_count),(6,6))
        updates=[u for e in r.events for u in e.distance_updates]
        self.assertEqual([u for u in updates if u.position==A],
                         [DistanceUpdate(A,None,10),DistanceUpdate(A,10,3)])
        self.assertEqual([u for u in updates if u.position==G],
                         [DistanceUpdate(G,None,32),DistanceUpdate(G,32,24)])
        self.assertEqual(r.events[2].newly_discovered,(G,))
        self.assertEqual(r.events[4].newly_discovered,())

    def test_compatibility(self):
        self.assertEqual(SearchEvent(S,()).current_cost,None)
        for e in bfs(self.demo()).events:
            self.assertIsNone(e.current_cost)
            self.assertEqual(e.distance_updates,())

    def test_comparison(self):
        maps=[GridMap([[0]*3 for _ in range(3)],S,(2,2)),self.demo(),
              GridMap([[0,1,0],[0,1,0],[0,0,0]],S,(0,2)),GridMap([[0,1,0]],S,(0,2))]
        for m in maps:
            a,b=bfs(m),dijkstra(m)
            self.assertEqual((a.success,a.path,a.cost,a.path_steps,a.discovered_count,a.expanded_count),
                             (b.success,b.path,b.cost,b.path_steps,b.discovered_count,b.expanded_count))
            self.assertEqual([(e.current,e.newly_discovered) for e in a.events],
                             [(e.current,e.newly_discovered) for e in b.events])

    def test_repeat_immutable_map(self):
        m=self.demo(); before=(m.grid,m.start,m.goal)
        self.assertEqual(dijkstra(m),dijkstra(m))
        self.assertEqual((m.grid,m.start,m.goal),before)

    def test_invalid(self):
        for value in (None,[],{},1,'map'):
            with self.assertRaisesRegex(TypeError,'GridMap'): dijkstra(value)

    def test_immutable_result(self):
        r=dijkstra(self.demo()); e=r.events[0]; u=e.distance_updates[0]
        for obj,field in ((r,'cost'),(e,'current_cost'),(u,'new_cost')):
            with self.assertRaises(FrozenInstanceError): setattr(obj,field,999)
        for seq in (r.path,r.events,e.current,e.newly_discovered,e.distance_updates,u.position):
            with self.assertRaises(TypeError): seq[0]=None


if __name__=='__main__': unittest.main()
