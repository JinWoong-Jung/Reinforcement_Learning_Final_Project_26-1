from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from agents.heuristic_agents import (
    HEURISTIC_POLICIES,
    heuristic_action,
)
from env.exam_env import ExamStrategyEnv
from utils.io import save_json, save_results_csv


HeuristicSelector = Callable[[ExamStrategyEnv], int]

HEURISTIC_MAP: dict[str, HeuristicSelector] = dict(HEURISTIC_POLICIES)


@dataclass
class EpisodeRecord:
    episode: int
    student_level: str
    total_reward: float
    total_score: float
    solved_count: int
    remaining_time_sec: float
    time_spent_total: float
    problem_time_spent: list[float]
    score_timeline: list[float]
    used_time_timeline: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "student_level": self.student_level,
            "total_reward": self.total_reward,
            "total_score": self.total_score,
            "solved_count": self.solved_count,
            "remaining_time_sec": self.remaining_time_sec,
            "time_spent_total": self.time_spent_total,
            "problem_time_spent": self.problem_time_spent,
            "score_timeline": self.score_timeline,
            "used_time_timeline": self.used_time_timeline,
        }


def _episode_metrics(env: ExamStrategyEnv, ep_reward: float, episode: int, student_level: str) -> EpisodeRecord:
    assert env.state is not None
    total_time = 0.0
    problem_times: list[float] = []
    for i, progress in enumerate(env.state.progress):
        spent = float(progress.time_spent_sec)
        problem_times.append(spent)
        total_time += spent

    return EpisodeRecord(
        episode=episode,
        student_level=student_level,
        total_reward=float(ep_reward),
        total_score=float(env.state.total_score),
        solved_count=int(env.state.solved_count()),
        remaining_time_sec=float(env.state.remaining_time_sec),
        time_spent_total=float(total_time),
        problem_time_spent=problem_times,
        score_timeline=[],
        used_time_timeline=[],
    )


def evaluate_policy(
    config: dict[str, Any],
    policy_name: str,
    episodes: int = 50,
    student_levels: tuple[str, ...] | None = None,
    student_id: str | None = None,
    student_level: str | None = None,
    rl_model: Any | None = None,
    rl_algorithm: str = "ppo",
    seed: int = 42,
) -> dict[str, Any]:
    if policy_name not in HEURISTIC_MAP and rl_model is None:
        raise ValueError("For non-heuristic policy_name, provide rl_model.")

    records: list[EpisodeRecord] = []
    for ep in range(episodes):
        ep_student_level = "mixed"
        reset_options: dict[str, Any] = {}
        if student_id is not None:
            reset_options["student_id"] = student_id
            ep_student_level = f"id:{student_id}"
        elif student_level is not None:
            reset_options["student_level"] = student_level
            ep_student_level = student_level
        elif student_levels:
            ep_level = student_levels[ep % len(student_levels)]
            reset_options["student_level"] = ep_level
            ep_student_level = ep_level

        base_env = ExamStrategyEnv(config=config, random_seed=seed + ep)
        is_dqn = rl_model is not None and rl_algorithm.lower() == "dqn"
        obs, _ = base_env.reset(seed=seed + ep, options=reset_options)

        done = False
        truncated = False
        ep_reward = 0.0
        score_timeline = [0.0]
        used_time_timeline = [0.0]
        while not (done or truncated):
            if rl_model is not None:
                raw_action, _ = rl_model.predict(obs, deterministic=True)
                action = raw_action
            else:
                action = heuristic_action(base_env, policy_name)

            obs, reward, done, truncated, info = base_env.step(action)
            ep_reward += float(reward)

            state = base_env.state
            assert state is not None
            score_timeline.append(float(state.total_score))
            used_time_timeline.append(float(base_env.total_time_sec - state.remaining_time_sec))

        record = _episode_metrics(base_env, ep_reward, ep, ep_student_level)
        record.score_timeline = score_timeline
        record.used_time_timeline = used_time_timeline
        records.append(record)

    summaries = _build_summary(records, policy_name)
    return {
        "policy_name": policy_name,
        "episodes": episodes,
        "mode": "rl" if rl_model is not None else "heuristic",
        "algorithm": rl_algorithm if rl_model is not None else None,
        "summary": summaries["overall"],
        "student_level_breakdown": summaries["by_level"],
        "problem_avg_time": summaries["problem_avg_time"],
        "episode_records": [r.to_dict() for r in records],
    }


def _build_summary(records: list[EpisodeRecord], policy_name: str) -> dict[str, Any]:
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    overall = {
        "policy_name": policy_name,
        "mean_score": _mean([r.total_score for r in records]),
        "mean_reward": _mean([r.total_reward for r in records]),
    }

    by_level: dict[str, dict[str, float]] = {}
    for level in sorted({r.student_level for r in records}):
        chunk = [r for r in records if r.student_level == level]
        by_level[level] = {
            "mean_score": _mean([r.total_score for r in chunk]),
            "mean_reward": _mean([r.total_reward for r in chunk]),
        }

    num_problems = len(records[0].problem_time_spent) if records else 0
    problem_avg_time = []
    for i in range(num_problems):
        problem_avg_time.append(_mean([r.problem_time_spent[i] for r in records]))

    return {"overall": overall, "by_level": by_level, "problem_avg_time": problem_avg_time}


def evaluate_heuristics_table(
    config: dict[str, Any],
    episodes: int = 60,
    seed: int = 42,
    student_id: str | None = None,
    student_level: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name in HEURISTIC_MAP:
        result = evaluate_policy(
            config=config,
            policy_name=policy_name,
            episodes=episodes,
            student_id=student_id,
            student_level=student_level,
            seed=seed,
        )
        row = {"policy_name": policy_name}
        row.update(result["summary"])
        rows.append(row)
    return rows


def save_results_json(results: dict[str, Any], path: str) -> None:
    save_json(results, path, indent=2)


def save_table_csv(rows: list[dict[str, Any]], path: str) -> None:
    save_results_csv(rows, path)
