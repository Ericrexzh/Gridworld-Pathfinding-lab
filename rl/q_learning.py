"""表格Q-learning:从reset/step采样,不查询MDP"""
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from envs import GridWorldEnv


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, low, high, include_low=True, include_high=True):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or (value < low if include_low else value <= low)
            or (value > high if include_high else value >= high)):
        raise ValueError(f"invalid {name}: {value!r}")


def _validate_q(Q, shape=None):
    if (not isinstance(Q, np.ndarray) or Q.ndim != 2 or min(Q.shape) == 0
            or not np.issubdtype(Q.dtype, np.floating) or not np.isfinite(Q).all()):
        raise ValueError("Q must be a nonempty finite 2D floating-point array")
    if shape is not None and Q.shape != tuple(shape):
        raise ValueError(f"Q shape must be {tuple(shape)}, got {Q.shape}")


def _index(value, size, name):
    _integer(value, name, 0)
    if value >= size:
        raise ValueError(f"{name} is out of bounds")


def epsilon_greedy(Q, state, epsilon, rng):
    """epsilon探索全部动作,利用时argmax优先最小编号"""
    _validate_q(Q)
    _index(state, Q.shape[0], "state")
    _number(epsilon, "epsilon", 0, 1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


def q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma):
    """原地更新一个Q单元并返回新值,外部截断不视为terminated"""
    _validate_q(Q)
    if not Q.flags.writeable:
        raise ValueError("Q must be writeable for updates")
    _index(state, Q.shape[0], "state")
    _index(next_state, Q.shape[0], "next_state")
    _index(action, Q.shape[1], "action")
    _number(alpha, "alpha", 0, 1, include_low=False)
    _number(gamma, "gamma", 0, 1, include_high=False)
    if isinstance(reward, bool) or not isinstance(reward, Real) or not np.isfinite(reward):
        raise ValueError("reward must be finite and real")
    if not isinstance(terminated, (bool, np.bool_)):
        raise ValueError("terminated must be boolean")
    target = float(reward) if terminated else float(reward) + gamma * float(np.max(Q[next_state]))
    updated = (1 - alpha) * float(Q[state, action]) + alpha * target
    if not np.isfinite(updated) or abs(updated) > np.finfo(Q.dtype).max:
        raise ArithmeticError("Q update overflow")
    Q[state, action] = updated
    return float(Q[state, action])


def train_q_learning(env, episodes=3000, alpha=0.1, gamma=0.9,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                     max_steps=100, seed=42):
    """返回 (Q, history),history的epsilon是该回合实际使用值
    环境有上限时必须与max_steps相同，无上限时由训练器保护
    history.truncated包含环境截断和训练器保护上限，两者均保留bootstrap
    """
    if not isinstance(env, GridWorldEnv):
        raise TypeError("env must be a GridWorldEnv")
    _integer(episodes, "episodes")
    _integer(max_steps, "max_steps")
    _integer(seed, "seed", 0)
    _number(alpha, "alpha", 0, 1, include_low=False)
    _number(gamma, "gamma", 0, 1, include_high=False)
    _number(epsilon_start, "epsilon_start", 0, 1)
    _number(epsilon_min, "epsilon_min", 0, epsilon_start)
    _number(epsilon_decay, "epsilon_decay", 0, 1, include_low=False)
    if env.max_episode_steps is not None and env.max_episode_steps != max_steps:
        raise ValueError("env.max_episode_steps must equal max_steps or be None")
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    history = []
    epsilon = float(epsilon_start)
    for episode in range(episodes):
        state, _ = env.reset(seed=int(seed) if episode == 0 else None)
        total_reward = 0.0
        for steps in range(1, max_steps + 1):
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, terminated, env_truncated, info = env.step(action)
            q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma)
            total_reward += reward
            state = next_state
            truncated = bool(not terminated and (env_truncated or steps == max_steps))
            if terminated or truncated:
                break
        history.append({"episode": episode + 1, "total_reward": total_reward,
                        "steps": steps, "success": bool(info["is_success"]),
                        "epsilon": epsilon, "terminated": bool(terminated),
                        "truncated": truncated})
        epsilon = max(float(epsilon_min), epsilon * epsilon_decay)
    return Q, history


def save_q_table(path, Q):
    """保存为NPZ,键名Q;不保存需要pickle的对象"""
    _validate_q(Q)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, Q=Q)


def load_q_table(path, expected_shape=None):
    """加载并验证;expected_shape用来拒绝不匹配的地图尺寸，当前实验并未体现"""
    with np.load(path, allow_pickle=False) as data:
        if "Q" not in data:
            raise ValueError("NPZ must contain Q")
        Q = data["Q"].copy()
    _validate_q(Q, expected_shape)
    return Q
