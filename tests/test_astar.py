"""真实 GridMap 专项图、堆监视和 128 张小图穷举。"""
import unittest
from pathlib import Path
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import search
from grid import GridMap
from search import a_star_search as astar, bfs, dijkstra, greedy_best_first_search as greedy


def relaxation_map():
    return GridMap([[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,0],
                    [0,1,0,0,0],[0,0,0,1,0]],(0,0),(4,4))


def stale_map():
    return GridMap([[0,0,0,0,0,0,0],[0,1,0,0,0,1,0],[1,0,0,0,0,0,0],
                    [0,0,1,0,0,1,1],[0,0,0,0,1,0,0],[0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0]],(0,0),(6,6))


def greedy_map():
    return GridMap([[0,0,0,0,1],[0,0,0,0,1],[0,1,0,1,0],[0,0,0,0,0]],(2,0),(2,4))


class AStarTests(unittest.TestCase):
    def maps(self):
        return [GridMap([[0,0]],(0,0),(0,1)),GridMap([[0]*3 for _ in range(3)],(0,0),(2,2)),
                GridMap.from_json(Path(__file__).resolve().parents[1]/'maps/demo.json'),
                greedy_map(),relaxation_map(),stale_map(),GridMap([[0,1,0]],(0,0),(0,2))]

    def check_path(self,m,r):
        self.assertEqual((r.path[0],r.path[-1]),(m.start,m.goal))
        self.assertEqual(len(set(r.path)),len(r.path))
        self.assertEqual(r.path_steps,len(r.path)-1)
        self.assertEqual(r.cost,sum(m.move_cost(a,b,'4') for a,b in zip(r.path,r.path[1:])))

    def test_direct(self):
        r=astar(self.maps()[0]); self.assertEqual((r.success,r.path_steps,r.cost),(True,1,10))

    def test_row(self):
        r=astar(GridMap([[0]*5],(0,0),(0,4)))
        self.assertEqual((r.path_steps,r.cost),(4,40))

    def test_square_stable(self):
        r=astar(GridMap([[0,0],[0,0]],(0,0),(1,1)))
        self.assertEqual(r.path,((0,0),(0,1),(1,1)))
        self.assertEqual(tuple(e.current for e in r.events),((0,0),(0,1),(1,0),(1,1)))
        self.assertEqual(r.cost,20)

    def test_demo(self):
        r=astar(self.maps()[2]); self.assertEqual((r.path_steps,r.cost),(8,80))

    def test_greedy_counterexample(self):
        m=greedy_map()
        self.assertEqual([f(m).cost for f in (bfs,dijkstra,greedy,astar)],[60,60,80,60])
        self.check_path(m,astar(m))

    def test_failure(self):
        r=astar(self.maps()[-1])
        self.assertEqual((r.success,r.path,r.cost,r.path_steps),(False,(),None,None))
        self.assertEqual((len(r.events),r.events[0].current_cost,r.events[0].current_heuristic),(1,0,20))

    def test_heap_and_event_replay(self):
        m=stale_map(); pushed=[]; popped=[]; calls=[]
        push,pop,neighbors=search.heapq.heappush,search.heapq.heappop,GridMap.neighbors
        def push_record(heap,item): pushed.append(item); return push(heap,item)
        def pop_record(heap):
            # 稳定顺序必须遵守 (f, sequence)，不允许 g 或坐标抢先。
            expected=min(heap,key=lambda item:(item[0],item[1]))
            item=pop(heap); self.assertEqual(item,expected); popped.append(item); return item
        def query(instance,pos,mode):
            self.assertEqual(mode,'4'); calls.append(pos); return neighbors(instance,pos,mode)
        with patch.object(search.heapq,'heappush',push_record),patch.object(search.heapq,'heappop',pop_record),patch.object(GridMap,'neighbors',query):
            r=astar(m)
        self.assertEqual([item[1] for item in pushed],list(range(len(pushed))))
        for f,seq,g,p in pushed:
            self.assertEqual(f,g+10*(abs(p[0]-6)+abs(p[1]-6)))
        dist={m.start:0}; settled=set(); events=iter(r.events); stale=[]
        for f,seq,g,p in popped:
            if g!=dist[p]: stale.append((p,g,dist[p])); continue
            self.assertNotIn(p,settled); settled.add(p)
            e=next(events)
            self.assertEqual((e.current,e.current_cost),(p,g))
            self.assertEqual(e.current_heuristic,m.heuristic(p,m.goal,'4'))
            self.assertEqual(f,e.current_cost+e.current_heuristic)
            first=[]
            for u in e.distance_updates:
                self.assertEqual(u.previous_cost,dist.get(u.position))
                self.assertEqual(u.new_cost,g+10)
                if u.previous_cost is None: first.append(u.position)
                else: self.assertLess(u.new_cost,u.previous_cost)
                dist[u.position]=u.new_cost
            self.assertEqual(e.newly_discovered,tuple(first))
        self.assertIsNone(next(events,None))
        self.assertGreaterEqual(len(stale),3)
        for expected in [((4,1),90,70),((5,1),100,80),((6,1),110,90)]: self.assertIn(expected,stale)
        self.assertEqual(calls,[e.current for e in r.events[:-1]])
        self.assertEqual(len(calls),len(set(calls)))
        self.assertEqual((r.discovered_count,r.expanded_count),(len(dist),len(settled)))
        self.assertEqual(r.discovered_count,1+sum(len(e.newly_discovered) for e in r.events))
        self.assertLessEqual(r.expanded_count,r.discovered_count)
        self.assertEqual(r.events[-1].newly_discovered,())
        self.assertEqual(r.events[-1].distance_updates,())
        self.assertEqual(r.cost,160)

    def test_relaxation_parent(self):
        m=relaxation_map(); saved_parent={}; original=search._reconstruct_path
        def capture(parent,goal):
            saved_parent.update(parent)
            return original(parent,goal)
        with patch.object(search,'_reconstruct_path',capture): r=astar(m)
        updates=[(e,u) for e in r.events for u in e.distance_updates if u.position==(2,4)]
        self.assertEqual([(u.previous_cost,u.new_cost) for e,u in updates],[(None,100),(100,80)])
        self.assertEqual(updates[-1][0].current,(1,4))
        self.assertNotIn((2,4),updates[-1][0].newly_discovered)
        self.assertEqual(saved_parent[(2,4)],(1,4))
        self.assertEqual(r.cost,100)
        self.check_path(m,r)

    def test_consistency_admissibility(self):
        for m in self.maps():
            self.assertEqual(m.heuristic(m.goal,m.goal,'4'),0)
            for row in range(m.height):
                for col in range(m.width):
                    p=(row,col)
                    if not m.is_walkable(p): continue
                    h=m.heuristic(p,m.goal,'4')
                    for q,cost in m.neighbors(p,'4'):
                        self.assertLessEqual(h,cost+m.heuristic(q,m.goal,'4'))
                    if p==m.goal: continue  # GridMap 禁止起终点相同；该点已独立检查0
                    remaining=bfs(GridMap(m.grid,p,m.goal))
                    if remaining.success: self.assertLessEqual(h,remaining.cost)
                    # 不可达时真实距离为无穷大，不用 None 做数值比较。

    def test_optimality_comparison(self):
        for m in self.maps():
            with self.subTest(size=(m.height,m.width),goal=m.goal):
                r=astar(m)
                for algorithm in (bfs,dijkstra):
                    ref=algorithm(m)
                    self.assertEqual((r.success,r.cost,r.path_steps),(ref.success,ref.cost,ref.path_steps))
                if r.success: self.check_path(m,r)

    def test_exhaustive_128(self):
        cells=[(r,c) for r in range(3) for c in range(3) if (r,c) not in ((0,0),(2,2))]
        for mask in range(128):
            grid=[[0]*3 for _ in range(3)]
            for i,(r,c) in enumerate(cells): grid[r][c]=(mask>>i)&1
            m=GridMap(grid,(0,0),(2,2))
            with self.subTest(mask=mask):
                result=astar(m)
                for algorithm in (bfs,dijkstra):
                    ref=algorithm(m); self.assertEqual((result.success,result.cost),(ref.success,ref.cost))
                if result.success: self.check_path(m,result)
                else: self.assertEqual((result.path,result.cost,result.path_steps),((),None,None))

    def test_repeat_immutable_map(self):
        m=stale_map(); before=(m.grid,m.start,m.goal)
        self.assertEqual(astar(m),astar(m))
        self.assertEqual((m.grid,m.start,m.goal),before)

    def test_compatibility(self):
        m=self.maps()[2]
        self.assertTrue(all(e.current_heuristic is None for e in bfs(m).events))
        self.assertTrue(all(e.current_heuristic is None for e in dijkstra(m).events))
        self.assertTrue(all(e.current_cost is None and not e.distance_updates for e in greedy(m).events))
        self.assertTrue(all(e.current_cost is not None and e.current_heuristic is not None for e in astar(m).events))

    def test_invalid_and_immutable(self):
        for value in (None,[],{},1,'map'):
            with self.assertRaisesRegex(TypeError,'GridMap'): astar(value)
        r=astar(self.maps()[0]); e=r.events[0]; u=e.distance_updates[0]
        for obj,field in ((r,'cost'),(e,'current_cost'),(u,'new_cost')):
            with self.assertRaises(FrozenInstanceError): setattr(obj,field,0)
        for seq in (r.path,r.events,e.current,e.newly_discovered,e.distance_updates,u.position):
            with self.assertRaises(TypeError): seq[0]=None


if __name__=='__main__': unittest.main()
