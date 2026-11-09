"""使用标准库 unittest；各测试的 docstring 说明验证性质。"""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from grid import GridMap


class GridMapTests(unittest.TestCase):
    def load_data(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return GridMap.from_json(path)

    def test_demo(self):
        """演示 JSON 的尺寸、地形和起终点完整保留。"""
        m = GridMap.from_json(Path(__file__).resolve().parents[1] / "maps/demo.json")
        self.assertEqual((m.height, m.width), (5, 5))
        self.assertEqual((m.start, m.goal), ((0, 0), (4, 4)))
        self.assertEqual(m.grid, ((0,0,0,0,0),(0,1,1,0,0),(0,0,0,0,0),(0,1,0,1,0),(0,0,0,0,0)))

    def test_invalid_grids(self):
        """拒绝空、非矩形、非法容器、未知编码及伪整数。"""
        cases = [(None, TypeError), ([], ValueError), ([[]], ValueError),
                 ([[0, 0], [0]], ValueError), ([0], TypeError),
                 ([[0, 2]], ValueError), ([[0, -1]], ValueError)]
        cases += [([[0, value]], TypeError) for value in (True, False, 0.0, "0", None)]
        for grid, error in cases:
            with self.subTest(grid=grid), self.assertRaises(error):
                GridMap(grid, (0, 0), (0, 1))

    def test_endpoints(self):
        """分别拒绝两个端点越界、落在障碍及重合。"""
        for start, goal, message in [((-1,0),(1,1), "start"),
                                     ((0,0),(2,0), "goal"),
                                     ((0,1),(1,1), "start"),
                                     ((0,0),(0,1), "goal"),
                                     ((0,0),(0,0), "different")]:
            with self.subTest(start=start, goal=goal), self.assertRaisesRegex(ValueError, message):
                GridMap([[0,1],[0,0]], start, goal)

    def test_coordinate_types(self):
        """坐标须有两个严格整数，构造和查询遵守同一约定。"""
        m = GridMap([[0,0]], (0,0), (0,1))
        for value, error in [(None,TypeError), ("01",TypeError),
                             ([0],ValueError), ([0,0,0],ValueError),
                             ([True,0],TypeError), ([0,0.0],TypeError)]:
            for field in ("start", "goal"):
                args = dict(grid=[[0,0]], start=(0,0), goal=(0,1))
                args[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(error):
                    GridMap(**args)
            for query in (m.in_bounds, m.is_walkable):
                with self.subTest(query=query.__name__, value=value), self.assertRaises(error):
                    query(value)

    def test_queries(self):
        """边界内可通行与障碍不同；四侧越界均安全返回 False。"""
        m = GridMap([[0,1,0],[0,0,0]], (0,0), (1,2))
        for position in ((0,0), [1,2]):
            self.assertTrue(m.in_bounds(position))
            self.assertTrue(m.is_walkable(position))
        self.assertTrue(m.in_bounds((0,1)))
        self.assertFalse(m.is_walkable((0,1)))
        for position in ((-1,0),(0,-1),(2,0),(0,3)):
            with self.subTest(position=position):
                self.assertFalse(m.in_bounds(position))
                self.assertFalse(m.is_walkable(position))

    def test_immutable(self):
        """外部输入修改、字段赋值、嵌套元素赋值均不能改变地图。"""
        grid, start, goal = [[0,0],[0,0]], [0,0], [1,1]
        m = GridMap(grid, start, goal)
        grid[0][0] = 1
        grid.append([1,1])
        start[0] = 1
        goal[0] = 0
        self.assertEqual(m.grid, ((0,0),(0,0)))
        self.assertEqual((m.start,m.goal), ((0,0),(1,1)))
        for field in ("grid", "start", "goal", "height", "width"):
            with self.subTest(field=field), self.assertRaises((FrozenInstanceError, TypeError, AttributeError)):
                setattr(m, field, None)
        with self.assertRaises(TypeError):
            m.grid[0][0] = 1
        with self.assertRaises(TypeError):
            m.start[0] = 1
        with self.assertRaises(TypeError):
            m.goal[0] = 0

    def test_disconnected(self):
        """隔墙导致不可达不影响格式合法性，允许从 JSON 加载。"""
        m = self.load_data(dict(grid=[[0,1,0]], start=[0,0], goal=[0,2]))
        self.assertEqual(m.width, 3)

    def test_single_column(self):
        """合法矩形不必是正方形，也允许只有一列。"""
        m = GridMap([[0],[0]], (0,0), (1,0))
        self.assertEqual((m.height,m.width), (2,1))

    def test_json_structure(self):
        """JSON 顶层必须是对象且包含三个必要字段。"""
        for data in ([], None, 1):
            with self.subTest(data=data), self.assertRaises(TypeError):
                self.load_data(data)
        for field in ("grid", "start", "goal"):
            data = dict(grid=[[0,0]], start=[0,0], goal=[0,1])
            del data[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.load_data(data)

    def test_json_validation(self):
        """文件加载必须经过构造校验，JSON 布尔值不能绕过规则。"""
        with self.assertRaises(TypeError):
            self.load_data(dict(grid=[[0,True]], start=[0,0], goal=[0,1]))

    def test_file_errors(self):
        """文件不存在和 JSON 语法错误保留可区分的标准异常。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            with self.assertRaises(FileNotFoundError):
                GridMap.from_json(path)
            path.write_text('{"grid":', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                GridMap.from_json(path)


if __name__ == "__main__":
    unittest.main()
