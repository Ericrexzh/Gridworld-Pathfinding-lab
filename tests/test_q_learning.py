"""Q-learning 手算、交互训练、截断语义和持久化。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grid import GridMap
from envs import GridWorldEnv
from rl.q_learning import (epsilon_greedy, q_learning_update, train_q_learning,
                           save_q_table, load_q_table)


def small_env(**kwargs):
    return GridWorldEnv(GridMap([[0,0,0]],(0,0),(0,2)), **kwargs)


class QLearningTests(unittest.TestCase):
    def test_shape_and_model_free(self):
        env = small_env()
        with patch.object(env,'model_transition',side_effect=AssertionError('model used')):
            Q, history = train_q_learning(env,episodes=3,max_steps=5)
        self.assertEqual(Q.shape,(3,4))
        self.assertEqual(Q.dtype,np.float64)
        self.assertEqual(len(history),3)
        np.testing.assert_array_equal(Q[2],np.zeros(4))

    def test_exploration_reproducible(self):
        Q = np.array([[0.,100.,0.,0.]])
        def actions():
            rng = np.random.default_rng(42)
            return [epsilon_greedy(Q,0,1.0,rng) for _ in range(100)]
        a = actions()
        self.assertEqual(a,actions())
        self.assertEqual(set(a),{0,1,2,3})

    def test_greedy_and_tie(self):
        rng = np.random.default_rng(42)
        Q = np.array([[1.,3.,2.,0.],[4.,4.,4.,4.]])
        for _ in range(10):
            self.assertEqual(epsilon_greedy(Q,0,0,rng),1)
            self.assertEqual(epsilon_greedy(Q,1,0,rng),0)

    def test_update_hand_calculation(self):
        Q = np.array([[2.,0.,0.,0.],[1.,4.,3.,2.]])
        before = Q.copy()
        # target=-1+.9*4=2.6; new=2+.1*(2.6-2)=2.06
        self.assertAlmostEqual(q_learning_update(Q,0,0,-1,1,False,.1,.9),2.06)
        before[0,0] = 2.06
        np.testing.assert_allclose(Q,before,rtol=0,atol=1e-12)

    def test_terminated_no_bootstrap(self):
        Q = np.zeros((2,4)); Q[1] = 100
        self.assertEqual(q_learning_update(Q,0,1,10,1,True,.1,.9),1.0)
        np.testing.assert_array_equal(Q[1],np.full(4,100.))

    def test_truncated_training_still_bootstraps(self):
        for limit in (None,2):
            # 正奖励自环使第二步的下一状态max Q非零，区分错误的截断清零。
            env = GridWorldEnv(GridMap([[0,1,0]],(0,0),(0,2)),
                               step_reward=1,max_episode_steps=limit)
            with patch('rl.q_learning.epsilon_greedy',return_value=0):
                Q, history = train_q_learning(env,episodes=1,max_steps=2,alpha=.5,gamma=.9)
            self.assertAlmostEqual(Q[0,0],.975)  # 第一步.5；第二步 .5*.5+.5*(1+.9*.5)
            self.assertEqual(history[0]['steps'],2)
            self.assertTrue(history[0]['truncated'])
            self.assertFalse(history[0]['terminated'])
            self.assertFalse(history[0]['success'])
            np.testing.assert_array_equal(Q[1:],np.zeros((2,4)))

    def test_success_at_limit(self):
        env = GridWorldEnv(GridMap([[0,0]],(0,0),(0,1)),max_episode_steps=1)
        with patch('rl.q_learning.epsilon_greedy',return_value=1):
            Q,h = train_q_learning(env,episodes=1,max_steps=1)
        self.assertEqual(Q[0,1],1.)
        self.assertTrue(h[0]['success'])
        self.assertTrue(h[0]['terminated'])
        self.assertFalse(h[0]['truncated'])

    def test_reproducibility(self):
        q1,h1 = train_q_learning(small_env(),episodes=30,max_steps=10,seed=7)
        q2,h2 = train_q_learning(small_env(),episodes=30,max_steps=10,seed=7)
        np.testing.assert_array_equal(q1,q2)
        self.assertEqual(h1,h2)
        self.assertTrue(all(1<=h['steps']<=10 for h in h1))
        self.assertEqual(h1[0]['epsilon'],1.)
        self.assertEqual(h1[1]['epsilon'],.995)

    def test_epsilon_floor(self):
        _,history = train_q_learning(small_env(),episodes=3,max_steps=1,
                                     epsilon_start=.2,epsilon_min=.1,epsilon_decay=.5)
        self.assertEqual([h['epsilon'] for h in history],[.2,.1,.1])

    def test_training_validation(self):
        for kwargs in ({'episodes':0},{'episodes':True},{'alpha':0},{'alpha':1.1},
                       {'gamma':1},{'gamma':-.1},{'gamma':float('nan')},
                       {'epsilon_start':-1},{'epsilon_min':1.1},{'epsilon_decay':0},
                       {'epsilon_decay':1.1},{'max_steps':0},{'max_steps':1.5},
                       {'seed':-1},{'seed':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                train_q_learning(small_env(),**kwargs)
        with self.assertRaisesRegex(ValueError,'max_episode_steps'):
            train_q_learning(small_env(max_episode_steps=2),episodes=1,max_steps=3)

    def test_update_and_selection_validation(self):
        Q = np.zeros((3,4))
        rng = np.random.default_rng(0)
        for bad in (np.zeros((3,4),dtype=int),np.zeros(3),np.full((3,4),np.nan),np.zeros((0,4))):
            with self.assertRaises(ValueError): epsilon_greedy(bad,0,.1,rng)
        for state in (-1,3,True,1.0):
            with self.assertRaises(ValueError): epsilon_greedy(Q,state,.1,rng)
        for epsilon in (-.1,1.1,float('inf'),True):
            with self.assertRaises(ValueError): epsilon_greedy(Q,0,epsilon,rng)
        with self.assertRaises(TypeError): epsilon_greedy(Q,0,.1,None)
        for kwargs in ({'action':4},{'next_state':3},{'terminated':1},{'reward':float('nan')}):
            args = dict(state=0,action=0,reward=-1,next_state=1,terminated=False,alpha=.1,gamma=.9)
            args.update(kwargs)
            with self.assertRaises(ValueError): q_learning_update(Q,**args)
        np.testing.assert_array_equal(Q,np.zeros((3,4)))

    def test_save_load(self):
        Q = np.arange(12,dtype=float).reshape(3,4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nested/q.npz'
            save_q_table(path,Q)
            loaded = load_q_table(path,(3,4))
            np.testing.assert_array_equal(Q,loaded)
            loaded[0,0] = -1
            self.assertEqual(Q[0,0],0)
            with self.assertRaises(ValueError): load_q_table(path,(4,4))
            with self.assertRaises(FileNotFoundError): load_q_table(Path(directory)/'missing.npz')
            np.savez(path,wrong=Q)
            with self.assertRaises(ValueError): load_q_table(path)


if __name__ == '__main__':
    unittest.main()
