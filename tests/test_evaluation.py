"""独立评估：轨迹、状态隔离、无学习以及有限退出。"""
import unittest
from unittest.mock import patch

import numpy as np

from envs import GridWorldEnv
from grid import GridMap
from rl.evaluation import evaluate_policy
from rl.q_learning import train_q_learning


class EvaluationTests(unittest.TestCase):
    def test_success_trajectory_and_isolation(self):
        env = GridWorldEnv(GridMap([[0,0,0]],(0,0),(0,2)))
        env.reset(); env.step(1)
        before = env._position,env._step_count,env._needs_reset
        Q = np.zeros((3,4)); Q[0,1]=Q[1,1]=2
        saved = Q.copy()
        Q.flags.writeable = False
        with patch.object(env,'reset',side_effect=AssertionError('original reset')), \
             patch.object(env,'step',side_effect=AssertionError('original step')), \
             patch.object(GridWorldEnv,'model_transition',side_effect=AssertionError('model used')):
            result = evaluate_policy(env,Q)
        self.assertEqual(result,dict(success=True,trajectory=((0,0),(0,1),(0,2)),
                                    states=(0,1,2),actions=(1,1),total_reward=9.,steps=2,
                                    terminated=True,truncated=False))
        np.testing.assert_array_equal(saved,Q)
        self.assertEqual((env._position,env._step_count,env._needs_reset),before)

    def test_unreachable_and_limits(self):
        for env_limit,guard,expected in ((None,3,3),(2,5,2),(5,2,2)):
            env = GridWorldEnv(GridMap([[0,1,0]],(0,0),(0,2)),max_episode_steps=env_limit)
            result = evaluate_policy(env,np.zeros((3,4)),max_steps=guard)
            self.assertFalse(result['success'])
            self.assertFalse(result['terminated'])
            self.assertTrue(result['truncated'])
            self.assertEqual(result['steps'],expected)
            self.assertEqual(result['total_reward'],-expected)
            self.assertEqual(result['trajectory'],((0,0),)*(expected+1))
            self.assertEqual(result['actions'],(0,)*expected)

    def test_goal_priority_and_custom_rewards(self):
        env = GridWorldEnv(GridMap([[0,0]],(0,0),(0,1)),goal_reward=7,max_episode_steps=1)
        Q = np.zeros((2,4)); Q[0,1]=1
        result = evaluate_policy(env,Q,max_steps=1)
        self.assertEqual((result['steps'],result['total_reward'],result['success']), (1,7.,True))
        self.assertFalse(result['truncated'])

    def test_validation(self):
        env = GridWorldEnv(GridMap([[0,0]],(0,0),(0,1)))
        for Q in (np.zeros((3,4)),np.zeros((2,8)),np.full((2,4),np.inf)):
            with self.assertRaises(ValueError): evaluate_policy(env,Q)
        for limit in (0,-1,True,1.0):
            with self.assertRaises(ValueError): evaluate_policy(env,np.zeros((2,4)),limit)

    def test_small_training_then_evaluation(self):
        env = GridWorldEnv(GridMap([[0,0,0]],(0,0),(0,2)))
        Q,_ = train_q_learning(env,episodes=100,max_steps=10,seed=42)
        result = evaluate_policy(env,Q,max_steps=10)
        self.assertTrue(result['success'])
        self.assertEqual(result['steps'],2)
        self.assertEqual(len(result['trajectory']),len(result['actions'])+1)


if __name__ == '__main__':
    unittest.main()
