"""Round 9：模型无副作用、Bellman手算、同步更新与单次改进。"""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from envs import GridWorldEnv
from grid import GridMap
from rl import get_valid_states, policy_evaluation, policy_improvement
from rl import dynamic_programming as dp


def demo_env(**kwargs):
    return GridWorldEnv(GridMap.from_json(Path(__file__).resolve().parents[1] / 'maps/demo.json'), **kwargs)


def line_env(**kwargs):
    return GridWorldEnv(GridMap([[0,0,0]], (0,0), (0,2)), **kwargs)


class DynamicProgrammingTests(unittest.TestCase):
    def test_t01_states(self):
        env = demo_env()
        expected = tuple(s for s in range(25) if s not in (6,7,16,18))
        self.assertEqual(get_valid_states(env), expected)
        self.assertIn(24, expected)
        self.assertEqual(len(set(expected)),len(expected))
        rectangle = GridWorldEnv(GridMap([[0,1,0],[0,0,0]],(0,0),(1,2)))
        self.assertEqual(get_valid_states(rectangle), (0,2,3,4,5))

    def test_t02_model(self):
        env = GridWorldEnv(GridMap([[0]*3 for _ in range(3)],(0,0),(2,2)))
        for action, target in enumerate((1,5,7,3)):
            self.assertEqual(env.model_transition(4,action), (target,-1.0,False,False))
        self.assertEqual(env.model_transition(0,0), (0,-1.0,False,True))
        self.assertEqual(demo_env().model_transition(1,2), (1,-1.0,False,True))
        self.assertEqual(env.model_transition(7,1), (8,10.0,True,False))
        custom = line_env(step_reward=-2,goal_reward=7)
        self.assertEqual(custom.model_transition(0,1), (1,-2.0,False,False))
        self.assertEqual(custom.model_transition(1,1), (2,7.0,True,False))

    def test_t03_no_side_effects(self):
        env = demo_env(max_episode_steps=1)
        for phase in ('before_reset','active','truncated'):
            if phase == 'active': env.reset(seed=42)
            if phase == 'truncated': env.step(0)
            before = copy.deepcopy(vars(env))
            with patch.object(env,'reset',side_effect=AssertionError('reset called')), \
                 patch.object(env,'step',side_effect=AssertionError('step called')):
                for _ in range(3):
                    self.assertEqual(env.model_transition(23,1), (24,10.0,True,False))
                    self.assertEqual(env.model_transition(0,0), (0,-1.0,False,True))
            for name in ('_position','_step_count','_needs_reset','grid_map','_np_random_seed'):
                self.assertEqual(getattr(env,name,None), before.get(name))
            if env._np_random is not None:
                self.assertEqual(env._np_random.bit_generator.state,
                                 before['_np_random'].bit_generator.state)
        # 成功结束后同样可以查询另一个非终止状态。
        env = line_env()
        env.reset(); env.step(1); env.step(1)
        self.assertEqual(env.model_transition(0,1), (1,-1.0,False,False))
        self.assertEqual((env._position,env._step_count,env._needs_reset), ((0,2),2,True))

    def test_t04_model_step_consistency(self):
        for env in (demo_env(),line_env(step_reward=-2,goal_reward=7)):
            m = env.grid_map
            goal = m.goal[0]*m.width+m.goal[1]
            for state in get_valid_states(env):
                if state == goal: continue
                local = GridMap(m.grid,divmod(state,m.width),m.goal)
                interactive = GridWorldEnv(local,step_reward=env.step_reward,
                                           goal_reward=env.goal_reward,max_episode_steps=1)
                for action in range(4):
                    interactive.reset()
                    obs,reward,term,trunc,info = interactive.step(action)
                    self.assertEqual(env.model_transition(state,action),
                                     (obs,reward,term,info['invalid_action']))
                    self.assertEqual(trunc,not term)

    def test_t05_parameters(self):
        env, policy = line_env(), {0:1,1:1}
        self.assertEqual(policy_evaluation(env,policy,gamma=0)[0], {0:-1.0,1:10.0,2:0.0})
        self.assertTrue(policy_evaluation(env,policy,gamma=.9)[1]['converged'])
        for gamma in (-.1,1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): policy_evaluation(env,policy,gamma=gamma)
            with self.assertRaises(ValueError): policy_improvement(env,{0:0,1:0,2:0},gamma=gamma)
        for theta in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): policy_evaluation(env,policy,theta=theta)
        for limit in (0,-1,True,1.0,'2',None):
            with self.assertRaises(ValueError): policy_evaluation(env,policy,max_iterations=limit)
        for field in ('gamma','theta'):
            for value in (True,'0.9',None):
                with self.assertRaises(TypeError): policy_evaluation(env,policy,**{field:value})

    def test_t06_bellman(self):
        values,info = policy_evaluation(line_env(),{0:1,1:1})
        self.assertEqual(values,{0:8.0,1:10.0,2:0.0})
        self.assertEqual(info,dict(iterations=3,delta=0.0,converged=True))

    def test_t07_self_loop(self):
        values,info = policy_evaluation(line_env(),{0:0,1:0})
        self.assertTrue(info['converged'])
        for state in (0,1):
            self.assertAlmostEqual(values[state],-10.0,delta=1e-7)
        self.assertLess(info['delta'],1e-8)
        # max_episode_steps不属于基础MDP，不能改变评估。
        self.assertEqual(policy_evaluation(line_env(max_episode_steps=1),{0:0,1:0}), (values,info))

    def test_t08_terminal(self):
        env = line_env()
        with self.assertRaisesRegex(ValueError,'terminal'): env.model_transition(2,0)
        values,_ = policy_evaluation(env,{0:1,1:1})
        self.assertEqual(values[2],0)
        self.assertEqual(values[1],10)
        self.assertNotIn(2,policy_improvement(env,values)[0])
        with self.assertRaises(ValueError): policy_evaluation(env,{0:1,1:1,2:0})

    def test_t09_improvement(self):
        old = {0:0,1:0}
        values = {0:8.0,1:10.0,2:0.0}
        snapshot = values.copy(),old.copy()
        result = policy_improvement(line_env(),values,old_policy=old)
        self.assertEqual(result,({0:1,1:1},(0,1)))
        self.assertEqual((values,old),snapshot)
        self.assertEqual(policy_improvement(line_env(),values),({0:1,1:1},()))
        self.assertEqual(policy_improvement(line_env(),values,old_policy=result[0])[1],())

    def test_t10_ties_all_actions(self):
        env = line_env()
        values = {0:0.0,1:0.0,2:0.0}
        self.assertEqual(policy_improvement(env,values)[0],{0:0,1:1})
        calls = []
        original = env.model_transition
        def record(s,a):
            calls.append((s,a)); return original(s,a)
        with patch.object(env,'model_transition',record):
            first = policy_improvement(env,values)
        self.assertEqual(calls,[(s,a) for s in (0,1) for a in range(4)])
        self.assertEqual(first,policy_improvement(env,values))
        # 一个严格更优、但差异很小的动作不能被任意容差吞掉。
        self.assertEqual(policy_improvement(env,{0:0,1:1e-10,2:0})[0][0],1)

    def test_t11_obstacles(self):
        env = demo_env()
        policy = {s:0 for s in get_valid_states(env) if s!=24}
        values,_ = policy_evaluation(env,policy)
        new,_ = policy_improvement(env,values)
        self.assertEqual(set(values),set(get_valid_states(env)))
        self.assertEqual(set(new),set(policy))
        for s in policy:
            for a in range(4):
                self.assertIn(env.model_transition(s,a)[0],values)

    def test_t12_policy_validation(self):
        for bad in ({0:0},{0:4,1:0},{0:True,1:0},{0:1.0,1:0},
                    {0:0,1:0,2:0},{False:0,1:0},{0.0:0,1:0},[],None):
            with self.subTest(policy=bad),self.assertRaises((TypeError,ValueError)):
                policy_evaluation(line_env(),bad)
        env = demo_env()
        policy = {s:0 for s in get_valid_states(env) if s!=24}
        policy[6] = 0
        with self.assertRaises(ValueError): policy_evaluation(env,policy)
        with self.assertRaises(ValueError):
            policy_improvement(line_env(),{0:0,1:0,2:0},old_policy={0:0})

    def test_t13_synchronous(self):
        # 倒序状态遍历会让原地更新把10立即传播给起点；同步第一轮必须是-1。
        for order in ((0,1,2),(2,1,0)):
            with patch.object(dp,'get_valid_states',return_value=order):
                v,info = policy_evaluation(line_env(),{0:1,1:1},max_iterations=1)
            self.assertEqual(v,{0:-1.0,1:10.0,2:0.0})
            self.assertEqual(info,dict(iterations=1,delta=10.0,converged=False))
        v,info = policy_evaluation(line_env(),{0:1,1:1},max_iterations=2)
        self.assertEqual(v[0],8)
        self.assertEqual(info,dict(iterations=2,delta=9.0,converged=False))

    def test_model_validation(self):
        env = demo_env()
        for s in (-1,25,6,24):
            with self.assertRaises(ValueError): env.model_transition(s,0)
        for s in (True,False,1.0,'1',None,np.bool_(True)):
            with self.assertRaises(TypeError): env.model_transition(s,0)
        for a in (-1,4,True,None,1.0):
            with self.assertRaises(ValueError): env.model_transition(0,a)
        self.assertEqual(env.model_transition(np.int64(0),np.int64(1))[0],1)

    def test_values_validation(self):
        env = line_env()
        for bad in ({0:0,1:0},{0:0,1:0,2:1},{0:0,1:0,2:0,3:0},
                    {False:0,1:0,2:0},{0:float('nan'),1:0,2:0},
                    {0:float('inf'),1:0,2:0},{0:True,1:0,2:0},[],None):
            with self.subTest(values=bad),self.assertRaises((TypeError,ValueError)):
                policy_improvement(env,bad)

    def test_convergence_boundary(self):
        _,info = policy_evaluation(line_env(),{0:0,1:0},theta=1.0,max_iterations=1)
        self.assertEqual(info,dict(iterations=1,delta=1.0,converged=False))
        _,info = policy_evaluation(line_env(),{0:0,1:0},theta=1.01,max_iterations=1)
        self.assertTrue(info['converged'])

    def test_dp_no_interaction(self):
        env = line_env()
        env.reset(seed=42); env.step(1)
        before = env._position,env._step_count,env._needs_reset
        policy = {0:0,1:0}
        with patch.object(env,'reset',side_effect=AssertionError), \
             patch.object(env,'step',side_effect=AssertionError):
            values,_ = policy_evaluation(env,policy)
            policy_improvement(env,values,old_policy=policy)
        self.assertEqual((env._position,env._step_count,env._needs_reset),before)
        self.assertEqual(policy,{0:0,1:0})


if __name__ == '__main__':
    unittest.main()
