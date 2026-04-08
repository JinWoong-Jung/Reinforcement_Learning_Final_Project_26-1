import os

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.makedirs("/tmp/mplconfig", exist_ok=True)

import json
import numpy as np
from stable_baselines3 import PPO

from env.exam_env import ExamStrategyEnv
from utils.io import load_config


def main() -> None:
    run_dir = os.environ.get("RUN_DIR", "runs/ppo_20260408_191834")
    cfg = load_config(f"{run_dir}/config_snapshot.yaml")
    model = PPO.load(f"{run_dir}/checkpoints/ppo_final.zip")

    env = ExamStrategyEnv(cfg, exam_data_path="data/25_math_calculus.json", student_data_path="data/someone.json")
    obs, info = env.reset(seed=0, options={"student_id": "Student"})
    traj = []
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        prev_idx = env.state.current_problem_idx
        obs, reward, done, truncated, step_info = env.step(action)
        traj.append(
            {
                "prev_problem_idx": prev_idx + 1,
                "action": action.tolist() if hasattr(action, "tolist") else list(action),
                "action_name": step_info["action_name"],
                "target_problem_idx": step_info["target_problem_idx"] + 1,
                "remaining_time_sec": step_info["remaining_time_sec"],
                "reward": round(float(reward), 4),
            }
        )

    state = env.state
    times = [
        (i + 1, round(p.time_spent_sec, 1), round(p.confidence_score, 4), p.status.value)
        for i, p in enumerate(state.progress)
        if p.time_spent_sec > 0
    ]
    times_sorted = sorted(times, key=lambda x: x[1], reverse=True)
    print("single_episode")
    print("start_problem", info["start_problem_idx"] + 1)
    print("final_score", round(state.total_score, 4), "solved", state.solved_count(), "steps", state.step_count)
    print("visit_order", [idx + 1 for idx in state.visit_order])
    print("top_time_spent", json.dumps(times_sorted[:10], ensure_ascii=False))
    print("first_30_steps", json.dumps(traj[:30], ensure_ascii=False))

    problem_time_totals = None
    problem_visit_counts = None
    problem_solved_counts = None
    for ep in range(20):
        env = ExamStrategyEnv(cfg, exam_data_path="data/25_math_calculus.json", student_data_path="data/someone.json")
        obs, _ = env.reset(seed=ep, options={"student_id": "Student"})
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)
        spent = np.array([p.time_spent_sec for p in env.state.progress], dtype=float)
        visited = np.array([1.0 if p.status.value != "NOT_VISITED" else 0.0 for p in env.state.progress], dtype=float)
        solved = np.array([1.0 if p.confidence_score >= 0.5 else 0.0 for p in env.state.progress], dtype=float)
        if problem_time_totals is None:
            problem_time_totals = spent
            problem_visit_counts = visited
            problem_solved_counts = solved
        else:
            problem_time_totals += spent
            problem_visit_counts += visited
            problem_solved_counts += solved
    avg = problem_time_totals / 20
    vis = problem_visit_counts / 20
    sol = problem_solved_counts / 20
    ranking = sorted(
        [(i + 1, round(float(avg[i]), 1), round(float(vis[i]), 2), round(float(sol[i]), 2)) for i in range(len(avg))],
        key=lambda x: x[1],
        reverse=True,
    )
    print("avg_time_top10", json.dumps(ranking[:10], ensure_ascii=False))


if __name__ == "__main__":
    main()
