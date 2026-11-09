"""冻结Q表的独立贪心轨迹评估"""
import numpy as np

from envs import GridWorldEnv
from .q_learning import _integer, _validate_q


def evaluate_policy(env, Q, max_steps=100):
    """用同地图、同奖励的新环境评估,传入env的episode不受影响。
    trajectory 包含起点及每步之后的(row,col),states是对应整数编码。
    无效移动保留重复位置，保证len(trajectory) == len(actions) + 1
    """
    if not isinstance(env, GridWorldEnv):
        raise TypeError("env must be a GridWorldEnv")
    _integer(max_steps, "max_steps")
    _validate_q(Q, (env.observation_space.n, env.action_space.n))
    evaluation_env = GridWorldEnv(env.grid_map, step_reward=env.step_reward,
                                 goal_reward=env.goal_reward,
                                 max_episode_steps=env.max_episode_steps)
    try:
        state, info = evaluation_env.reset()
        states, trajectory, actions = [state], [info["position"]], []
        total_reward = 0.0
        for steps in range(1, max_steps + 1):
            action = int(np.argmax(Q[state]))
            state, reward, terminated, env_truncated, info = evaluation_env.step(action)
            actions.append(action)
            states.append(state)
            trajectory.append(info["position"])
            total_reward += reward
            truncated = bool(not terminated and (env_truncated or steps == max_steps))
            if terminated or truncated:
                break
        return {"success": bool(info["is_success"]), "trajectory": tuple(trajectory),
                "states": tuple(states), "actions": tuple(actions),
                "total_reward": total_reward, "steps": steps,
                "terminated": bool(terminated), "truncated": truncated}
    finally:
        evaluation_env.close()
