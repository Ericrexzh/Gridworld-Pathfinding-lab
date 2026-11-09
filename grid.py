"""静态地图、移动规则与几何启发式；不包含搜索算法。"""

import json
from dataclasses import dataclass
from pathlib import Path

ORTHOGONAL_COST = 10
DIAGONAL_COST = 14.14
DIRECTIONS_4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECTIONS_8 = ((-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1))


def _directions(mode):
    """统一校验模式，返回有序且不可变的方向偏移。"""
    if not isinstance(mode, str) or mode not in ("4", "8"):
        raise ValueError("mode must be '4' or '8'")
    return DIRECTIONS_4 if mode == "4" else DIRECTIONS_8


def _edge_cost(source, target):
    """仅供已经确认合法的单步移动使用。"""
    diagonal = source[0] != target[0] and source[1] != target[1]
    return DIAGONAL_COST if diagonal else ORTHOGONAL_COST


def _position(value, name):
    """校验坐标结构并复制为不可变元组。"""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of two integers")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    if any(type(item) is not int for item in value):
        raise TypeError(f"{name} coordinates must be integers (not bool)")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class GridMap:
    """校验后不可变的地图；构造时也接受列表并复制为元组。"""

    grid: tuple[tuple[int, ...], ...]
    start: tuple[int, int]
    goal: tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.grid, (list, tuple)):
            raise TypeError("grid must be a list or tuple of rows")
        if not self.grid:
            raise ValueError("grid must not be empty")
        rows = []
        for index, row in enumerate(self.grid):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"grid row {index} must be a list or tuple")
            if not row:
                raise ValueError(f"grid row {index} must not be empty")
            if rows and len(row) != len(rows[0]):
                raise ValueError(f"grid row {index} has inconsistent width")
            for col, cell in enumerate(row):
                if type(cell) is not int:
                    raise TypeError(f"grid[{index}][{col}] must be an integer (not bool)")
                if cell not in (0, 1):
                    raise ValueError(f"grid[{index}][{col}] must be 0 or 1")
            rows.append(tuple(row))
        # frozen 阻止普通赋值；这里只在初始化阶段写入校验后的副本。
        object.__setattr__(self, "grid", tuple(rows))
        object.__setattr__(self, "start", _position(self.start, "start"))
        object.__setattr__(self, "goal", _position(self.goal, "goal"))
        for name in ("start", "goal"):
            position = getattr(self, name)
            if not self.in_bounds(position):
                raise ValueError(f"{name} {position} is out of bounds")
            if not self.is_walkable(position):
                raise ValueError(f"{name} {position} is on an obstacle")
        if self.start == self.goal:
            raise ValueError("start and goal must be different")

    @classmethod
    def from_json(cls, path: str | Path):
        """读取 UTF-8 JSON；文件和 JSON 语法错误保留标准异常。"""
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError("map JSON must be an object")
        missing = {"grid", "start", "goal"} - data.keys()
        if missing:
            raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
        return cls(data["grid"], data["start"], data["goal"])

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    def in_bounds(self, position) -> bool:
        """格式错误抛异常；格式正确但越界返回 False。"""
        row, col = _position(position, "position")
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, position) -> bool:
        """越界或障碍返回 False；不会使用 Python 的负索引绕回地图。"""
        if not self.in_bounds(position):
            return False
        row, col = position
        return self.grid[row][col] == 0

    def can_move(self, source, target, mode) -> bool:
        """判断单步移动；格式错误抛异常，空间上不合法返回 False。"""
        directions = _directions(mode)
        source = _position(source, "source")
        target = _position(target, "target")
        return self._can_move(source, target, directions)

    def _can_move(self, source, target, directions) -> bool:
        """共享合法性判定；调用方已校验坐标格式与模式。"""
        dr, dc = target[0] - source[0], target[1] - source[1]
        if (dr, dc) not in directions:
            return False
        if not self.is_walkable(source) or not self.is_walkable(target):
            return False
        if dr != 0 and dc != 0:
            return (self.is_walkable((source[0] + dr, source[1]))
                    and self.is_walkable((source[0], source[1] + dc)))
        return True

    def move_cost(self, source, target, mode) -> int | float:
        """返回合法单步边权；非法移动抛 ValueError。"""
        if not self.can_move(source, target, mode):
            raise ValueError(f"illegal move from {source} to {target} in mode {mode}")
        return _edge_cost(source, target)

    def neighbors(self, position, mode) -> list:
        """按固定方向顺序返回 [(坐标, 代价)]；非法来源抛异常。"""
        directions = _directions(mode)
        position = _position(position, "position")
        if not self.is_walkable(position):
            raise ValueError(f"position {position} must be in bounds and walkable")
        result = []
        for dr, dc in directions:
            target = (position[0] + dr, position[1] + dc)
            if self._can_move(position, target, directions):
                result.append((target, _edge_cost(position, target)))
        return result

    def heuristic(self, position, goal, mode) -> int | float:
        """界内两点的几何下界；允许障碍坐标，不计绕行成本。"""
        _directions(mode)
        position = _position(position, "position")
        goal = _position(goal, "goal")
        if not self.in_bounds(position) or not self.in_bounds(goal):
            raise ValueError("position and goal must be in bounds")
        dr, dc = abs(position[0] - goal[0]), abs(position[1] - goal[1])
        if mode == "4":
            return ORTHOGONAL_COST * (dr + dc)
        return DIAGONAL_COST * min(dr, dc) + ORTHOGONAL_COST * abs(dr - dc)
