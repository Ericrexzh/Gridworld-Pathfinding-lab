"""运行python train.py默认训练demo,结果写入项目results目录"""
import argparse
import csv
import json
from pathlib import Path
import platform

import gymnasium
import numpy as np

from grid import GridMap
from envs import GridWorldEnv
from rl.q_learning import train_q_learning, save_q_table, load_q_table
from rl.evaluation import evaluate_policy


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=root / "maps/demo.json")
    parser.add_argument("--output", type=Path, default=root / "results")
    for name, default in (("episodes",3000),("max_steps",100),("seed",42)):
        parser.add_argument("--"+name.replace("_","-"), type=int, default=default)
    for name, default in (("alpha",.1),("gamma",.9),("epsilon_start",1.),
                          ("epsilon_min",.05),("epsilon_decay",.995)):
        parser.add_argument("--"+name.replace("_","-"), type=float, default=default)
    args = parser.parse_args()
    parameters = {k:v for k,v in vars(args).items() if k not in ("map","output")}
    grid_map = GridMap.from_json(args.map)
    env = GridWorldEnv(grid_map, max_episode_steps=args.max_steps)
    try:
        Q, history = train_q_learning(env, **parameters)
        evaluation = evaluate_policy(env, Q, max_steps=args.max_steps)
    finally:
        env.close()
    window = min(100,len(history))
    summary = {"training_successes": sum(h["success"] for h in history),
               "episodes":len(history), "final_window":window,
               "final_window_success_rate":sum(h["success"] for h in history[-window:])/window,
               "mean_total_reward":float(np.mean([h["total_reward"] for h in history])),
               "mean_steps":float(np.mean([h["steps"] for h in history])),
               "final_epsilon":history[-1]["epsilon"]}
    config = {"algorithm":"Q-learning", "parameters":parameters,
              "map_path":str(args.map.resolve()),
              "map":{"grid":grid_map.grid,"start":grid_map.start,"goal":grid_map.goal},
              "actions":["UP","RIGHT","DOWN","LEFT"],
              "rewards":{"step":env.step_reward,"goal":env.goal_reward},
              "versions":{"python":platform.python_version(),"numpy":np.__version__,
                          "gymnasium":gymnasium.__version__},
              "epsilon_schedule":"record before each episode; decay after episode with minimum floor",
              "truncation_bootstrap":True, "summary":summary, "evaluation":evaluation}
    args.output.mkdir(parents=True, exist_ok=True)
    save_q_table(args.output / "q_table.npz", Q)
    
    #实际验证持久化往返
    np.testing.assert_array_equal(load_q_table(args.output / "q_table.npz",Q.shape), Q)
    with (args.output / "training_history.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "training_config.json").write_text(
        json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Training summary (final window is last 100 episodes or fewer):")
    print(json.dumps(summary,indent=2))
    print("Independent greedy evaluation (one deterministic episode):")
    print(json.dumps(evaluation,indent=2))
    print("Saved:",args.output.resolve())


if __name__ == "__main__":
    main()
