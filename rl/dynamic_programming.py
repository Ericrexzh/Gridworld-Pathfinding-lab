"""确定性策略的同步评估与单次贪心改进,当前还不包含策略迭代(Policy Iteration)循环"""
from math import isfinite
from numbers import Integral, Real

from envs import GridWorldEnv


def get_valid_states(env):
    """返回行优先排序的全部可通行编码,包含终点"""
    if not isinstance(env, GridWorldEnv):
        raise TypeError("env must be a GridWorldEnv")
    m = env.grid_map
    return tuple(r * m.width + c for r in range(m.height) for c in range(m.width)
                 if m.is_walkable((r, c)))


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_gamma(gamma):
    _finite_real(gamma, "gamma")
    if not 0 <= gamma < 1:
        raise ValueError("gamma must satisfy 0 <= gamma < 1")


def _validate_keys(mapping, expected, name):
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dict")
    if any(isinstance(s, bool) or not isinstance(s, Integral) for s in mapping):
        raise TypeError(f"{name} state keys must be integers (not bool)")
    if set(mapping) != set(expected):
        raise ValueError(f"{name} must cover exactly the required states")


def _policy_model(env, policy, nonterminal):
    """校验完整策略,并缓存每个所选动作的纯模型转移"""
    _validate_keys(policy, nonterminal, "policy")
    return {s: env.model_transition(s, policy[s]) for s in nonterminal}


def _action_value(transition, values, gamma):
    next_state, reward, terminated, _invalid = transition
    value = reward if terminated else reward + gamma * values[next_state]
    if not isfinite(value):
        raise ArithmeticError("Bellman update produced a non-finite value")
    return value


def policy_evaluation(env, policy, gamma=0.9, theta=1e-8, max_iterations=10000):
    """从全零 V 同步迭代,返回 (values, iterations/delta/converged)"""
    states = get_valid_states(env)
    _validate_gamma(gamma)
    _finite_real(theta, "theta")
    if theta <= 0:
        raise ValueError("theta must be positive")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral)
            or max_iterations <= 0):
        raise ValueError("max_iterations must be a positive integer")
    goal = env.grid_map.goal[0] * env.grid_map.width + env.grid_map.goal[1]
    nonterminal = tuple(s for s in states if s != goal)
    model = _policy_model(env, policy, nonterminal)
    values = dict.fromkeys(states, 0.0)
    for iteration in range(1, max_iterations + 1):
        new_values = {s: _action_value(model[s], values, gamma) for s in nonterminal}
        new_values[goal] = 0.0
        delta = max(abs(new_values[s] - values[s]) for s in states)
        values = new_values
        if delta < theta:
            return values, {"iterations": iteration, "delta": delta, "converged": True}
    return values, {"iterations": iteration, "delta": delta, "converged": False}


def policy_improvement(env, values, gamma=0.9, *, old_policy=None):
    """只改进一次,同Q取小编号,没有旧策略时changed_states=()"""
    states = get_valid_states(env)
    _validate_gamma(gamma)
    _validate_keys(values, states, "values")
    for value in values.values():
        _finite_real(value, "value")
    goal = env.grid_map.goal[0] * env.grid_map.width + env.grid_map.goal[1]
    if values[goal] != 0:
        raise ValueError("values at terminal state must be zero")
    nonterminal = tuple(s for s in states if s != goal)
    if old_policy is not None:
        _policy_model(env, old_policy, nonterminal)
    new_policy = {}
    for s in nonterminal:
        best_action, best_value = 0, float("-inf")
        for action in range(env.action_space.n):
            q = _action_value(env.model_transition(s, action), values, gamma)
            if q > best_value:  
                best_action, best_value = action, q
        new_policy[s] = best_action
    changed = (() if old_policy is None else
               tuple(s for s in nonterminal if new_policy[s] != old_policy[s]))
    return new_policy, changed
