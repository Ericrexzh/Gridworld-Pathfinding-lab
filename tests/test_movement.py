"""Round 2：预期邻居与代价独立手写，浮点比较容差为 1e-9。"""
import unittest
from grid import GridMap


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.m = GridMap([[0]*3 for _ in range(3)], (0,0), (2,2))

    def test_t01_four_neighbors(self):
        self.assertEqual(self.m.neighbors((1,1), '4'),
                         [((0,1),10), ((1,2),10), ((2,1),10), ((1,0),10)])

    def test_t02_eight_neighbors(self):
        self.assertEqual(self.m.neighbors((1,1), '8'),
                         [((0,1),10), ((0,2),14.14), ((1,2),10), ((2,2),14.14),
                          ((2,1),10), ((2,0),14.14), ((1,0),10), ((0,0),14.14)])

    def test_t03_boundaries(self):
        # 手工枚举四角及四个边缘的八方向坐标；过滤正交项得四方向。
        cases = {
            (0,0): [(0,1),(1,1),(1,0)],
            (0,2): [(1,2),(1,1),(0,1)],
            (2,2): [(1,2),(2,1),(1,1)],
            (2,0): [(1,0),(1,1),(2,1)],
            (0,1): [(0,2),(1,2),(1,1),(1,0),(0,0)],
            (1,2): [(0,2),(2,2),(2,1),(1,1),(0,1)],
            (2,1): [(1,1),(1,2),(2,2),(2,0),(1,0)],
            (1,0): [(0,0),(0,1),(1,1),(2,1),(2,0)],
        }
        for source, positions in cases.items():
            for mode in ('4','8'):
                expected = []
                for target in positions:
                    diagonal = source[0] != target[0] and source[1] != target[1]
                    if mode == '8' or not diagonal:
                        expected.append((target, 14.14 if diagonal else 10))
                with self.subTest(source=source, mode=mode):
                    self.assertEqual(self.m.neighbors(source, mode), expected)

    def test_t04_obstacles(self):
        m = GridMap([[0,1,0],[0,0,0],[0,0,0]], (0,0), (2,2))
        self.assertEqual(m.neighbors((1,1),'4'), [((1,2),10),((2,1),10),((1,0),10)])

    def test_t05_corners(self):
        # 四个对角方向逐一测试五种情况，包括单侧阻挡。
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            target = (1+dr,1+dc)
            side1, side2 = (1+dr,1), (1,1+dc)
            for blocked in ([], [side1], [side2], [side1,side2], [target]):
                cells = [[0]*3 for _ in range(3)]
                for r,c in blocked:
                    cells[r][c] = 1
                endpoints = [(r,c) for r in range(3) for c in range(3) if cells[r][c] == 0]
                m = GridMap(cells, endpoints[0], endpoints[-1])
                with self.subTest(target=target, blocked=blocked):
                    self.assertEqual(m.can_move((1,1),target,'8'), not blocked)
                    self.assertEqual(target in [p for p,_ in m.neighbors((1,1),'8')], not blocked)
                    if blocked:
                        with self.assertRaises(ValueError):
                            m.move_cost((1,1),target,'8')

    def test_t06_orthogonal_cost(self):
        for target in ((0,1),(1,2),(2,1),(1,0)):
            self.assertEqual(self.m.move_cost((1,1),target,'4'), 10)

    def test_t07_eight_cost(self):
        for target, cost in [((0,1),10),((1,2),10),((2,1),10),((1,0),10),
                             ((0,0),14.14),((0,2),14.14),((2,0),14.14),((2,2),14.14)]:
            self.assertAlmostEqual(self.m.move_cost((1,1),target,'8'), cost, delta=1e-9)

    def test_t08_four_rejects_diagonal(self):
        for target in ((0,0),(0,2),(2,0),(2,2)):
            self.assertFalse(self.m.can_move((1,1),target,'4'))
            with self.assertRaises(ValueError):
                self.m.move_cost((1,1),target,'4')

    def test_t09_not_single_step(self):
        for mode in ('4','8'):
            for source,target in [((1,1),(1,1)),((0,0),(0,2)),((0,0),(2,2))]:
                self.assertFalse(self.m.can_move(source,target,mode))
                with self.assertRaises(ValueError):
                    self.m.move_cost(source,target,mode)

    def test_t10_invalid_inputs(self):
        m = GridMap([[0,1,0],[0,0,0],[0,0,0]], (0,0), (2,2))
        for mode in ('4','8'):
            for bad,error in [(None,TypeError),('12',TypeError),([0],ValueError),
                              ([0,1,2],ValueError),([True,1],TypeError),([0,1.0],TypeError)]:
                calls = [lambda: m.neighbors(bad,mode)]
                for method in (m.can_move,m.move_cost,m.heuristic):
                    calls.extend([lambda f=method: f(bad,(1,1),mode),
                                  lambda f=method: f((1,1),bad,mode)])
                for call in calls:
                    with self.subTest(bad=bad,mode=mode), self.assertRaises(error):
                        call()
            for invalid in ((-1,0),(3,0),(0,1)):
                self.assertFalse(m.can_move(invalid,(1,1),mode))
                self.assertFalse(m.can_move((1,1),invalid,mode))
                with self.assertRaises(ValueError):
                    m.neighbors(invalid,mode)
                with self.assertRaises(ValueError):
                    m.move_cost(invalid,(1,1),mode)
            for invalid in ((-1,0),(3,0)):
                with self.assertRaises(ValueError):
                    m.heuristic(invalid,(1,1),mode)
                with self.assertRaises(ValueError):
                    m.heuristic((1,1),invalid,mode)
        for mode in ('bad',4,8,None,True,[],{}):
            for call in (lambda: m.neighbors((1,1),mode),
                         lambda: m.can_move((1,1),(1,2),mode),
                         lambda: m.move_cost((1,1),(1,2),mode),
                         lambda: m.heuristic((1,1),(1,2),mode)):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    call()

    def test_t11_manhattan(self):
        m = GridMap([[0]*7 for _ in range(7)], (0,0),(6,6))
        self.assertEqual(m.heuristic((2,3),(6,5),'4'),60)

    def test_t12_octile(self):
        m = GridMap([[0]*7 for _ in range(7)], (0,0),(6,6))
        self.assertAlmostEqual(m.heuristic((2,3),(6,5),'8'),48.28,delta=1e-9)

    def test_t13_heuristic_edges(self):
        for mode in ('4','8'):
            for p,q,value in [((1,1),(1,1),0),((1,1),(0,1),10),
                               ((0,0),(2,0),20),((0,0),(0,2),20)]:
                self.assertEqual(self.m.heuristic(p,q,mode),value)
        self.assertEqual(self.m.heuristic((0,0),(1,1),'4'),20)
        self.assertAlmostEqual(self.m.heuristic((0,0),(1,1),'8'),14.14,delta=1e-9)
        m = GridMap([[0,1],[0,0]], (0,0),(1,1))
        self.assertEqual(m.heuristic((0,1),(1,1),'4'),10)  # 允许障碍几何距离

    def test_t14_immutable(self):
        before = (self.m.grid,self.m.start,self.m.goal)
        for mode in ('4','8'):
            self.m.can_move([1,1],[1,2],mode)
            self.m.move_cost([1,1],[1,2],mode)
            result = self.m.neighbors([1,1],mode)
            result.clear()  # 返回列表属于调用者，不是地图内部存储
            self.m.heuristic([1,1],[2,2],mode)
        self.assertEqual((self.m.grid,self.m.start,self.m.goal),before)

    def test_t15_consistency(self):
        maps = [self.m, GridMap([[0,1,0],[0,0,0],[0,1,0]],(0,0),(2,2)),
                GridMap([[0,1,0]],(0,0),(0,2))]
        for m in maps:
            nodes = [(r,c) for r in range(m.height) for c in range(m.width) if m.grid[r][c] == 0]
            for mode in ('4','8'):
                for goal in nodes:
                    for n in nodes:
                        for nxt,cost in m.neighbors(n,mode):
                            with self.subTest(mode=mode,n=n,nxt=nxt,goal=goal):
                                self.assertLessEqual(m.heuristic(n,goal,mode),
                                                     cost+m.heuristic(nxt,goal,mode)+1e-9)

    def test_isolated_source(self):
        m = GridMap([[0,1,0]],(0,0),(0,2))
        for mode in ('4','8'):
            self.assertEqual(m.neighbors((0,0),mode),[])


if __name__ == '__main__':
    unittest.main()
