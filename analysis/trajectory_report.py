from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dpo_model import DPOPolicyModel
from agents.heuristic_agents import heuristic_action
from agents.train_rl import _build_env, _load_obs_normalizer
from env.exam_env import ExamStrategyEnv
from env.state import solved_criteria_from_config
from utils.model_compat import build_sb3_custom_objects, install_numpy_pickle_compat
from utils.io import load_config, save_json

try:
    from stable_baselines3 import DQN, PPO
except ImportError:  # pragma: no cover
    DQN = None
    PPO = None


def _load_model(model_path: str, algorithm: str, config: dict[str, Any]):
    algo = algorithm.lower()
    if algo == "dpo":
        return DPOPolicyModel.load(model_path)
    install_numpy_pickle_compat()
    env = _build_env(config=config, for_dqn=algo == "dqn", seed=int(config.get("experiment", {}).get("seed", 42)))
    custom_objects = build_sb3_custom_objects(config, algo, env)
    if algo == "ppo":
        if PPO is None:
            raise ImportError("stable-baselines3 is required to load PPO models.")
        return PPO.load(model_path, env=env, custom_objects=custom_objects)
    if algo == "dqn":
        if DQN is None:
            raise ImportError("stable-baselines3 is required to load DQN models.")
        return DQN.load(model_path, env=env, custom_objects=custom_objects)
    raise ValueError("algorithm must be 'ppo', 'dqn', or 'dpo'")


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir:
        snapshot = os.path.join(args.run_dir, "config_snapshot.yaml")
        if os.path.exists(snapshot):
            cfg = load_config(snapshot)
        elif args.config:
            cfg = load_config(args.config)
        else:
            raise FileNotFoundError(f"config_snapshot.yaml not found in {args.run_dir}")
    elif args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config("configs/default.yaml")

    cfg.setdefault("data", {})
    if args.exam_data:
        cfg["data"]["exam_path"] = args.exam_data
        cfg["data"].pop("exam_paths", None)
    if args.student_data:
        cfg["data"]["student_path"] = args.student_data
    cfg.setdefault("student", {})
    if args.student_id:
        cfg["student"]["fixed_id"] = args.student_id
    if args.student_level:
        cfg["student"]["fixed_level"] = args.student_level
    return cfg


def _resolve_obs_stats_path(run_dir: str | None, model_path: str | None) -> str | None:
    candidates: list[str] = []
    if model_path:
        candidates.append(os.path.join(os.path.dirname(model_path), "obs_stats.npz"))
    if run_dir:
        candidates.append(os.path.join(run_dir, "checkpoints", "obs_stats.npz"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _topk_time_share(problem_time_spent: list[float], k: int) -> float:
    if not problem_time_spent:
        return 0.0
    total = float(sum(problem_time_spent))
    if total <= 0:
        return 0.0
    topk = sum(sorted((float(x) for x in problem_time_spent), reverse=True)[:k])
    return float(topk / total)


def _problem_snapshot(
    env: ExamStrategyEnv,
    problem_idx: int,
    solved_criteria: dict[str, float],
) -> dict[str, Any]:
    problem = env.problems[problem_idx]
    progress = env.state.progress[problem_idx]
    predicted_choice = progress.predicted_choice_index()
    return {
        "problem_idx": problem_idx + 1,
        "problem_type": problem.problem_type,
        "difficulty_level": problem.difficulty_level,
        "score": int(problem.score),
        "time_spent_sec": round(float(progress.time_spent_sec), 4),
        "status": progress.status.value,
        "confidence_score": round(float(progress.confidence_score), 4),
        "observable_confidence": round(float(progress.observable_confidence(problem)), 4),
        "effective_confidence": round(float(progress.effective_confidence(problem)), 4),
        "answer_confidence": round(float(progress.answer_confidence), 4),
        "choice_confidences": [round(float(x), 4) for x in progress.choice_confidences],
        "predicted_choice": (int(predicted_choice) + 1) if predicted_choice is not None else None,
        "correct_choice": (int(problem.correct_choice_index) + 1) if problem.correct_choice_index is not None else None,
        "is_solved": bool(progress.is_solved(problem, **solved_criteria)),
    }


def _type_breakdown(env: ExamStrategyEnv, solved_criteria: dict[str, float]) -> dict[str, Any]:
    objective = []
    subjective = []
    for idx in range(len(env.problems)):
        snapshot = _problem_snapshot(env, idx, solved_criteria)
        if snapshot["problem_type"] == "objective":
            objective.append(snapshot)
        else:
            subjective.append(snapshot)
    objective.sort(key=lambda x: x["time_spent_sec"], reverse=True)
    subjective.sort(key=lambda x: x["time_spent_sec"], reverse=True)
    return {
        "objective_top5_by_time": objective[:5],
        "subjective_top5_by_time": subjective[:5],
    }


def _run_episode(
    env: ExamStrategyEnv,
    *,
    rl_model: Any | None,
    obs_normalizer,
    policy_name: str | None,
    seed: int,
    reset_options: dict[str, Any],
    max_logged_steps: int,
    solved_criteria: dict[str, float],
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options=reset_options)
    if obs_normalizer is not None:
        obs = obs_normalizer(np.asarray(obs, dtype=np.float32))
    trajectory: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    truncated = False
    problem_entry_counts = {int(info["start_problem_idx"]): 1}
    problem_work_session_counts: dict[int, int] = {}
    deferred_hard_problems: dict[int, dict[str, Any]] = {}
    revisited_hard_events: list[dict[str, Any]] = []
    current_session_problem_idx = int(info["start_problem_idx"])
    current_session_had_work = False
    current_session_is_revisit = False
    current_session_pending_hard_revisit: dict[str, Any] | None = None

    def _finalize_session(problem_idx: int) -> None:
        nonlocal current_session_had_work
        if current_session_had_work:
            problem_work_session_counts[problem_idx] = problem_work_session_counts.get(problem_idx, 0) + 1
        current_session_had_work = False

    while not (done or truncated):
        prev_idx = env.state.current_problem_idx if env.state is not None else -1
        prev_problem = env.problems[prev_idx] if env.state is not None and prev_idx >= 0 else None
        prev_progress = env.state.progress[prev_idx] if env.state is not None and prev_idx >= 0 else None
        if rl_model is not None:
            action, _ = rl_model.predict(obs, deterministic=True)
        else:
            if policy_name is None:
                raise ValueError("policy_name is required for heuristic trajectory reports.")
            action = heuristic_action(env, policy_name)

        obs, reward, done, truncated, step_info = env.step(action)
        if obs_normalizer is not None:
            obs = obs_normalizer(np.asarray(obs, dtype=np.float32))
        total_reward += float(reward)

        if step_info["action_name"] == "solve_more" and prev_idx >= 0:
            current_session_had_work = True
            if current_session_is_revisit and current_session_pending_hard_revisit is not None:
                event = dict(current_session_pending_hard_revisit)
                event["return_step"] = len(trajectory)
                event["revisit_count"] = problem_work_session_counts.get(prev_idx, 0)
                revisited_hard_events.append(event)
                current_session_pending_hard_revisit = None

        if step_info["action_name"] == "next":
            new_idx = int(step_info["current_problem_idx"])
            _finalize_session(prev_idx)
            prev_problem = env.problems[prev_idx] if prev_idx >= 0 else None
            prev_observable_conf = (
                float(prev_progress.observable_confidence(prev_problem))
                if prev_problem is not None and prev_progress is not None
                else 0.0
            )
            if (
                prev_problem is not None
                and prev_problem.difficulty_level in {"상", "최상"}
                and prev_progress is not None
                and prev_progress.time_spent_sec > 0.0
                and prev_observable_conf < 0.45
            ):
                deferred_hard_problems[prev_idx] = {
                    "problem_idx": prev_idx + 1,
                    "difficulty_level": prev_problem.difficulty_level,
                    "left_with_observable_confidence": round(prev_observable_conf, 4),
                }
            problem_entry_counts[new_idx] = problem_entry_counts.get(new_idx, 0) + 1
            current_session_problem_idx = new_idx
            current_session_is_revisit = problem_work_session_counts.get(new_idx, 0) > 0
            current_session_pending_hard_revisit = (
                deferred_hard_problems.pop(new_idx, None) if current_session_is_revisit else None
            )

        if len(trajectory) < max_logged_steps:
            trajectory.append(
                {
                    "prev_problem_idx": prev_idx + 1,
                    "prev_problem_type": prev_problem.problem_type if prev_problem is not None else None,
                    "action": action.tolist() if hasattr(action, "tolist") else list(action),
                    "action_name": step_info["action_name"],
                    "target_problem_idx": step_info["target_problem_idx"] + 1,
                    "remaining_time_sec": float(step_info["remaining_time_sec"]),
                    "expected_score": float(step_info["expected_score"]),
                    "same_problem_streak": int(step_info["same_problem_streak"]),
                    "prev_observable_confidence": (
                        round(float(prev_progress.observable_confidence(prev_problem)), 4)
                        if prev_problem is not None and prev_progress is not None
                        else None
                    ),
                    "prev_effective_confidence": (
                        round(float(prev_progress.effective_confidence(prev_problem)), 4)
                        if prev_problem is not None and prev_progress is not None
                        else None
                    ),
                    "current_problem_idx": int(step_info["current_problem_idx"]) + 1,
                    "reward": float(reward),
                }
            )

    assert env.state is not None
    _finalize_session(current_session_problem_idx)
    problem_time_spent = [float(p.time_spent_sec) for p in env.state.progress]
    ranking = sorted(
        [_problem_snapshot(env, i, solved_criteria) for i in range(len(problem_time_spent))],
        key=lambda x: x["time_spent_sec"],
        reverse=True,
    )
    revisited_problem_indices = sorted([idx + 1 for idx, count in problem_work_session_counts.items() if count > 1])
    revisit_count = int(sum(max(count - 1, 0) for count in problem_work_session_counts.values()))
    for snapshot in ranking:
        idx = snapshot["problem_idx"] - 1
        snapshot["entry_count"] = int(problem_entry_counts.get(idx, 0))
        snapshot["visit_count"] = int(problem_work_session_counts.get(idx, 0))
        snapshot["was_revisited"] = bool(snapshot["visit_count"] > 1)

    return {
        "start_problem_idx": int(info["start_problem_idx"]) + 1,
        "student_id": info["student_id"],
        "exam_path": info["exam_path"],
        "total_reward": float(total_reward),
        "total_score": float(env.state.total_score),
        "solved_count": int(env.state.solved_count(env.problems, **solved_criteria)),
        "visited_count": int(env.state.visited_count()),
        "coverage_fraction": float(env.state.coverage_fraction()),
        "objective_dominance_rate": float(env.state.objective_dominance_rate(env.problems)),
        "mean_subjective_confidence": float(env.state.mean_subjective_confidence(env.problems)),
        "subjective_solved_rate": float(env.state.subjective_solved_rate(env.problems, **solved_criteria)),
        "objective_solved_rate": float(env.state.objective_solved_rate(env.problems, **solved_criteria)),
        "remaining_time_sec": float(env.state.remaining_time_sec),
        "steps": int(env.state.step_count),
        "visit_order": [idx + 1 for idx in env.state.visit_order],
        "revisit_count": revisit_count,
        "revisited_problem_indices": revisited_problem_indices,
        "revisited_hard_problem_events": revisited_hard_events,
        "top1_time_share": _topk_time_share(problem_time_spent, 1),
        "top2_time_share": _topk_time_share(problem_time_spent, 2),
        "problem_ranking": ranking,
        "type_breakdown": _type_breakdown(env, solved_criteria),
        "trajectory_head": trajectory,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    num_problems = len(reports[0]["problem_ranking"])
    avg_times = []
    for i in range(num_problems):
        avg_times.append(
            {
                "problem_idx": i + 1,
                "avg_time_spent_sec": float(np.mean([r["problem_ranking"][i]["time_spent_sec"] for r in reports])),
            }
        )
    avg_times.sort(key=lambda x: x["avg_time_spent_sec"], reverse=True)
    return {
        "episodes": len(reports),
        "mean_reward": float(np.mean([r["total_reward"] for r in reports])),
        "mean_score": float(np.mean([r["total_score"] for r in reports])),
        "mean_solved_count": float(np.mean([r["solved_count"] for r in reports])),
        "mean_visited_count": float(np.mean([r["visited_count"] for r in reports])),
        "mean_coverage_fraction": float(np.mean([r["coverage_fraction"] for r in reports])),
        "mean_objective_dominance_rate": float(np.mean([r["objective_dominance_rate"] for r in reports])),
        "mean_subjective_confidence": float(np.mean([r["mean_subjective_confidence"] for r in reports])),
        "mean_subjective_solved_rate": float(np.mean([r["subjective_solved_rate"] for r in reports])),
        "mean_objective_solved_rate": float(np.mean([r["objective_solved_rate"] for r in reports])),
        "mean_remaining_time_sec": float(np.mean([r["remaining_time_sec"] for r in reports])),
        "mean_steps": float(np.mean([r["steps"] for r in reports])),
        "mean_revisit_count": float(np.mean([r["revisit_count"] for r in reports])),
        "mean_top1_time_share": float(np.mean([r["top1_time_share"] for r in reports])),
        "mean_top2_time_share": float(np.mean([r["top2_time_share"] for r in reports])),
        "avg_time_top10": avg_times[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trajectory and coverage sanity reports.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing config_snapshot.yaml")
    parser.add_argument("--config", type=str, default=None, help="Config path if no run dir is provided")
    parser.add_argument("--model-path", type=str, default=None, help="Path to PPO/DQN model zip")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "dqn", "dpo"], default="ppo")
    parser.add_argument("--policy-name", type=str, default=None, help="Heuristic policy name")
    parser.add_argument("--exam-data", type=str, default=None)
    parser.add_argument("--student-data", type=str, default=None)
    parser.add_argument("--student-id", type=str, default=None)
    parser.add_argument("--student-level", type=str, choices=["low", "mid", "high"], default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-logged-steps", type=int, default=40)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _resolve_config(args)

    if bool(args.model_path) == bool(args.policy_name):
        raise ValueError("Provide exactly one of --model-path or --policy-name")

    rl_model = _load_model(args.model_path, args.algorithm, cfg) if args.model_path else None
    obs_stats_path = _resolve_obs_stats_path(args.run_dir, args.model_path) if rl_model is not None else None
    obs_normalizer = _load_obs_normalizer(obs_stats_path) if rl_model is not None else None
    solved_criteria = solved_criteria_from_config(cfg)
    reset_options: dict[str, Any] = {}
    if args.student_id:
        reset_options["student_id"] = args.student_id
    elif args.student_level:
        reset_options["student_level"] = args.student_level

    reports = []
    for ep in range(args.episodes):
        env = (
            _build_env(config=cfg, for_dqn=args.algorithm.lower() == "dqn", seed=args.seed + ep)
            if rl_model is not None
            else ExamStrategyEnv(config=cfg, random_seed=args.seed + ep)
        )
        reports.append(
            _run_episode(
                env,
                rl_model=rl_model,
                obs_normalizer=obs_normalizer,
                policy_name=args.policy_name,
                seed=args.seed + ep,
                reset_options=reset_options,
                max_logged_steps=args.max_logged_steps,
                solved_criteria=solved_criteria,
            )
        )

    result = {
        "mode": "rl" if rl_model is not None else "heuristic",
        "algorithm": args.algorithm if rl_model is not None else None,
        "policy_name": args.policy_name,
        "summary": _aggregate_reports(reports),
        "first_episode": reports[0] if reports else None,
    }

    if args.output:
        save_json(result, args.output, indent=2)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
