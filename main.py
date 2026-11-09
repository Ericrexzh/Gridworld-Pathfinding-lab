"""统一寻路演示入口,其中强化学习的Q-learning搜索只读取训练结果"""
import json
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

from grid import GridMap
from search import bfs, dijkstra, greedy_best_first_search, a_star_search
from envs import GridWorldEnv
from rl.q_learning import load_q_table
from rl.evaluation import evaluate_policy


def build_sessions(grid_map, results_dir):
    """预先计算五种搜索及一次独立评估,回放控制不再运行算法"""
    sessions = []
    algorithms = [("BFS",bfs), ("Dijkstra",dijkstra),
                  ("Greedy",greedy_best_first_search),
                  ("A* / 4 directions",lambda m: a_star_search(m,mode="4")),
                  ("A* / 8 directions",lambda m: a_star_search(m,mode="8"))]
    for name, algorithm in algorithms:
        start = perf_counter()
        result = algorithm(grid_map)
        sessions.append(dict(name=name,result=result,elapsed_ms=(perf_counter()-start)*1000,
                             error=None))
    env = GridWorldEnv(grid_map)
    try:
        results_dir = Path(results_dir)
        Q = load_q_table(results_dir / "q_table.npz",
                         (env.observation_space.n,env.action_space.n))
        config_path = results_dir / "training_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            expected = {"grid":[list(row) for row in grid_map.grid],
                        "start":list(grid_map.start),"goal":list(grid_map.goal)}
            if config.get("map") != expected:
                raise ValueError("Q-table training map does not match demo.json")
        result = evaluate_policy(env,Q,max_steps=100)
        sessions.append(dict(name="Q-learning",result=result,error=None))
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as error:
        sessions.append(dict(name="Q-learning",result=None,
                             error="Q-table unavailable. Check results/q_table.npz. " + str(error)))
    finally:
        env.close()
    return sessions


def main():
    import pygame
    from render import DemoApp
    root = Path(__file__).resolve().parent
    grid_map = GridMap.from_json(root / "maps/demo.json")
    pygame.display.init()
    pygame.font.init()
    try:
        app = DemoApp(grid_map,build_sessions(grid_map,root / "results"))
        app.run()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
