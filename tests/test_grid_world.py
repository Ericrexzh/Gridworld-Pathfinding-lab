"""Round 8：Gymnasium 接口、任务语义和共享移动规则。"""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils.env_checker import check_env

from envs import GridWorldEnv
from grid import GridMap


def demo_map():
    return GridMap.from_json(Path(__file__).resolve().parents[1] / 'maps/demo.json')


class GridWorldTests(unittest.TestCase):
    def test_t01_initialization(self):
        m = GridMap([[0]*4 for _ in range(2)], (1,2), (0,3))
        env = GridWorldEnv(m)
        self.assertIsInstance(env, Env)
        self.assertIsInstance(env.observation_space, spaces.Discrete)
        self.assertIsInstance(env.action_space, spaces.Discrete)
        self.assertEqual((env.observation_space.n,env.action_space.n), (8,4))
        self.assertIs(env.grid_map, m)
        self.assertEqual(env.reset()[0], 6)  # 用非方形地图发现误用 height 的编码。

    def test_t02_reset(self):
        env = GridWorldEnv(demo_map())
        obs, info = env.reset(seed=42)
        self.assertIs(type(obs), int)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info, dict(position=(0,0),goal=(4,4),step_count=0,
                                    is_success=False,invalid_action=False))
        env.step(1)
        self.assertEqual(env.reset(seed=42, options={}), (obs,info))

    def test_t03_actions(self):
        env = GridWorldEnv(demo_map())
        for a in range(4):
            self.assertTrue(env.action_space.contains(a))
            env.reset()
            self.assertEqual(len(env.step(a)), 5)
        for a in range(4,8):
            self.assertFalse(env.action_space.contains(a))
        env.reset()
        self.assertEqual(env.step(np.int64(1))[0], 1)
        env.reset()
        self.assertEqual(env.step(np.array(1, dtype=np.int64))[0], 1)

    def test_t04_directions(self):
        env = GridWorldEnv(GridMap([[0]*3 for _ in range(3)], (1,1), (2,2)))
        for action, target, obs in [(0,(0,1),1),(1,(1,2),5),(2,(2,1),7),(3,(1,0),3)]:
            with self.subTest(action=action):
                env.reset()
                result = env.step(action)
                self.assertEqual(result[:4], (obs,-1.0,False,False))
                self.assertEqual(result[4]['position'], target)
                self.assertFalse(result[4]['invalid_action'])
                self.assertEqual(result[4]['step_count'], 1)

    def test_t05_boundaries(self):
        for start,action in [((0,0),0),((0,0),3),((2,2),1),((2,2),2)]:
            env = GridWorldEnv(GridMap([[0]*3 for _ in range(3)],start,(1,1)))
            obs,_ = env.reset()
            result = env.step(action)
            self.assertEqual(result[:4], (obs,-1.0,False,False))
            self.assertEqual(result[4]['position'], start)
            self.assertTrue(result[4]['invalid_action'])
            self.assertEqual(result[4]['step_count'], 1)

    def test_t06_obstacle(self):
        m = demo_map()
        before = m.grid,m.start,m.goal
        env = GridWorldEnv(m)
        env.reset()
        env.step(1)
        obs,reward,term,trunc,info = env.step(2)
        self.assertEqual((obs,reward,term,trunc), (1,-1.0,False,False))
        self.assertEqual((info['position'],info['step_count'],info['invalid_action']), ((0,1),2,True))
        self.assertEqual((m.grid,m.start,m.goal), before)

    def test_t07_goal(self):
        env = GridWorldEnv(GridMap([[0,0,0]], (0,0),(0,2)))
        env.reset()
        self.assertEqual(env.step(1)[:4], (1,-1.0,False,False))
        result = env.step(1)
        self.assertEqual(result[:4], (2,10.0,True,False))
        self.assertTrue(result[4]['is_success'])
        self.assertFalse(result[4]['invalid_action'])

    def test_t08_truncation(self):
        env = GridWorldEnv(demo_map(), max_episode_steps=2)
        env.reset()
        self.assertEqual(env.step(0)[:4], (0,-1.0,False,False))
        result = env.step(0)  # 两次越界也消耗两个时间步。
        self.assertEqual(result[:4], (0,-1.0,False,True))
        self.assertFalse(result[4]['is_success'])
        self.assertEqual(result[4]['step_count'], 2)
        self.assertEqual(env.reset()[1]['step_count'], 0)
        self.assertFalse(env.step(1)[3])

    def test_t09_goal_priority(self):
        for length in (1,2):
            env = GridWorldEnv(GridMap([[0]*(length+1)], (0,0),(0,length)),
                               max_episode_steps=length)
            env.reset()
            for _ in range(length):
                result = env.step(1)
            self.assertEqual(result[1:4], (10.0,True,False))

    def test_t10_invalid_inputs(self):
        env = GridWorldEnv(demo_map())
        for action in (-1,4,8,1.0,'1',None,True,False,[],{},np.array([1]),np.bool_(True)):
            with self.subTest(action=action):
                env.reset()
                with self.assertRaisesRegex(ValueError, 'action'):
                    env.step(action)
                # 错误输入不消耗步数，不允许先 int(1.5) 再校验。
                result = env.step(1)
                self.assertEqual(result[0], 1)
                self.assertEqual(result[4]['step_count'], 1)

    def test_t11_requires_reset(self):
        goal_env = GridWorldEnv(GridMap([[0,0]], (0,0),(0,1)))
        truncated_env = GridWorldEnv(demo_map(), max_episode_steps=1)
        for env,action in ((goal_env,1),(truncated_env,0)):
            with self.assertRaisesRegex(RuntimeError,'reset'):
                env.step(action)
            env.reset()
            result = env.step(action)
            self.assertTrue(result[2] or result[3])
            with self.assertRaisesRegex(RuntimeError,'reset'):
                env.step(action)
            env.reset()
            self.assertEqual(env.step(action), result)

    def test_t12_determinism(self):
        m = demo_map()
        before = m.grid,m.start,m.goal
        env = GridWorldEnv(m)
        traces = []
        for seed in (42,42,7,None):
            trace = [env.reset(seed=seed)]
            trace.extend(env.step(a) for a in (1,2,1,1,1,2,2,2,2))
            traces.append(trace)
        for trace in traces[1:]:
            self.assertEqual(trace, traces[0])
        self.assertEqual((m.grid,m.start,m.goal), before)

    def test_t13_official_checker(self):
        # 无渲染，空间、reset、step、随机数初始化等检查全部保留。
        for limit in (None,10):
            env = GridWorldEnv(demo_map(), max_episode_steps=limit)
            check_env(env, skip_render_check=True)
            env.close()
            env.close()

    def test_t14_grid_consistency(self):
        """遍历每个非终点可通行状态及四动作；打乱邻居顺序仍应一致。"""
        original = GridMap.neighbors
        calls = []
        def reversed_neighbors(instance, position, mode):
            calls.append(mode)
            return list(reversed(original(instance,position,mode)))
        for m in (demo_map(), GridMap([[0,1,0],[0,0,0]], (0,0),(1,2))):
            for row in range(m.height):
                for col in range(m.width):
                    p = row,col
                    if not m.is_walkable(p) or p == m.goal:
                        continue
                    local = GridMap(m.grid,p,m.goal)
                    env = GridWorldEnv(local)
                    legal = dict(m.neighbors(p,'4'))
                    for a,(dr,dc) in enumerate(((-1,0),(0,1),(1,0),(0,-1))):
                        target = row+dr,col+dc
                        expected = target if target in legal else p
                        env.reset()
                        with patch.object(GridMap,'neighbors',reversed_neighbors):
                            obs,reward,term,trunc,info = env.step(a)
                        self.assertEqual(info['position'],expected)
                        self.assertEqual(obs,expected[0]*m.width+expected[1])
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertTrue(m.is_walkable(expected))
                        self.assertEqual(info['invalid_action'],target not in legal)
                        self.assertEqual(term,expected==m.goal)
                        self.assertEqual(reward,10.0 if term else -1.0)
                        self.assertFalse(trunc)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {'4'})

    def test_configuration(self):
        for m in (None,[],{},'map'):
            with self.assertRaises(TypeError):
                GridWorldEnv(m)
        for limit in (0,-1,True,1.5,'2'):
            with self.assertRaises(ValueError):
                GridWorldEnv(demo_map(), max_episode_steps=limit)
        for field in ('step_reward','goal_reward'):
            for value in (True,'1',None,complex(1,2)):
                with self.assertRaises(TypeError):
                    GridWorldEnv(demo_map(), **{field:value})
            for value in (float('inf'),float('-inf'),float('nan')):
                with self.assertRaises(ValueError):
                    GridWorldEnv(demo_map(), **{field:value})
        env = GridWorldEnv(GridMap([[0,0]],(0,0),(0,1)),step_reward=-2,goal_reward=7)
        env.reset()
        self.assertEqual(env.step(0)[1], -2.0)
        self.assertEqual(env.step(1)[1], 7.0)

    def test_no_limit_and_info_snapshot(self):
        env = GridWorldEnv(demo_map())
        initial = env.reset()[1]
        for _ in range(20):
            result = env.step(0)
            self.assertEqual(result[2:4], (False,False))
        self.assertEqual(initial['step_count'], 0)
        result[4]['position'] = (99,99)
        result[4]['step_count'] = -1
        next_result = env.step(1)
        self.assertEqual(next_result[4]['position'], (0,1))
        self.assertEqual(next_result[4]['step_count'], 21)

    def test_reset_options(self):
        env = GridWorldEnv(demo_map())
        for options in ({'start':(2,2)},[],1):
            with self.assertRaises(ValueError):
                env.reset(options=options)
        self.assertEqual(env.reset(options={})[0], 0)


if __name__ == '__main__':
    unittest.main()
