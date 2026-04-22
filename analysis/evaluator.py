from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from agents.heuristic_agents import (
    HEURISTIC_POLICIES,
    heuristic_action,
)
from agents.train_rl import _build_env
from env.exam_env import ExamStrategyEnv
from env.problem import Problem
from env.state import ExamState, solved_criteria_from_config
from utils.io import save_json, save_results_csv

from agents.train_rl import _load_obs_normalizer


HeuristicSelector = Callable[[ExamStrategyEnv], np.ndarray]

HEURISTIC_MAP: dict[str, HeuristicSelector] = dict(HEURISTIC_POLICIES)


def realized_score_rollout(
    state: ExamState,
    problems: list[Problem],
    n_rollouts: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Sample realized scores using Bernoulli draws per problem.

    Each rollout independently samples:
        outcome_i ~ Bernoulli(P_i(t_i))   where P_i = effective_confidence
        realized  = sum_i outcome_i * w_i

    Returns:
        (mean_realized_score, std_realized_score) over n_rollouts.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = np.array(
        [progress.effective_confidence(problem) for progress, problem in zip(state.progress, problems)],
        dtype=np.float64,
    )
    weights = np.array([float(problem.score) for problem in problems], dtype=np.float64)

    # (n_rollouts, n_problems) boolean matrix
    outcomes = rng.random((n_rollouts, len(problems))) < probs[np.newaxis, :]
    scores = (outcomes * weights[np.newaxis, :]).sum(axis=1)
    return float(np.mean(scores)), float(np.std(scores))


@dataclass
class EpisodeRecord:
    episode: int
    student_level: str
    total_reward: float
    total_score: float
    solved_count: int
    visited_count: int
    coverage_fraction: float
    objective_dominance_rate: float
    mean_subjective_confidence: float
    subjective_solved_rate: float
    objective_solved_rate: float
    top1_time_share: float
    top2_time_share: float
    remaining_time_sec: float
    steps: int
    time_spent_total: float
    problem_time_spent: list[float]
    problem_pids: list[int]
    problem_difficulty_levels: list[str]
    problem_scores: list[float]
    problem_types: list[str]
    visit_order: list[int]
    score_timeline: list[float]
    used_time_timeline: list[float]
    realized_score_mean: float = 0.0
    realized_score_std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "student_level": self.student_level,
            "total_reward": self.total_reward,
            "total_score": self.total_score,
            "realized_score_mean": self.realized_score_mean,
            "realized_score_std": self.realized_score_std,
            "solved_count": self.solved_count,
            "visited_count": self.visited_count,
            "coverage_fraction": self.coverage_fraction,
            "objective_dominance_rate": self.objective_dominance_rate,
            "mean_subjective_confidence": self.mean_subjective_confidence,
            "subjective_solved_rate": self.subjective_solved_rate,
            "objective_solved_rate": self.objective_solved_rate,
            "top1_time_share": self.top1_time_share,
            "top2_time_share": self.top2_time_share,
            "remaining_time_sec": self.remaining_time_sec,
            "steps": self.steps,
            "time_spent_total": self.time_spent_total,
            "problem_time_spent": self.problem_time_spent,
            "problem_pids": self.problem_pids,
            "problem_difficulty_levels": self.problem_difficulty_levels,
            "problem_scores": self.problem_scores,
            "problem_types": self.problem_types,
            "visit_order": self.visit_order,
            "score_timeline": self.score_timeline,
            "used_time_timeline": self.used_time_timeline,
        }


def _episode_metrics(
    env: ExamStrategyEnv,
    ep_reward: float,
    episode: int,
    student_level: str,
    solved_criteria: dict[str, float],
) -> EpisodeRecord:
    assert env.state is not None
    total_time = 0.0
    problem_times: list[float] = []
    for i, progress in enumerate(env.state.progress):
        spent = float(progress.time_spent_sec)
        problem_times.append(spent)
        total_time += spent
    sorted_times = sorted(problem_times, reverse=True)
    top1_time = sorted_times[0] if sorted_times else 0.0
    top2_time = sum(sorted_times[:2]) if sorted_times else 0.0
    top1_time_share = float(top1_time / total_time) if total_time > 0 else 0.0
    top2_time_share = float(top2_time / total_time) if total_time > 0 else 0.0

    return EpisodeRecord(
        episode=episode,
        student_level=student_level,
        total_reward=float(ep_reward),
        total_score=float(env.state.total_score),
        solved_count=int(env.state.solved_count(env.problems, **solved_criteria)),
        visited_count=int(env.state.visited_count()),
        coverage_fraction=float(env.state.coverage_fraction()),
        objective_dominance_rate=float(env.state.objective_dominance_rate(env.problems)),
        mean_subjective_confidence=float(env.state.mean_subjective_confidence(env.problems)),
        subjective_solved_rate=float(env.state.subjective_solved_rate(env.problems, **solved_criteria)),
        objective_solved_rate=float(env.state.objective_solved_rate(env.problems, **solved_criteria)),
        top1_time_share=top1_time_share,
        top2_time_share=top2_time_share,
        remaining_time_sec=float(env.state.remaining_time_sec),
        steps=int(env.state.step_count),
        time_spent_total=float(total_time),
        problem_time_spent=problem_times,
        problem_pids=[int(problem.pid) for problem in env.problems],
        problem_difficulty_levels=[str(problem.difficulty_level) for problem in env.problems],
        problem_scores=[float(problem.score) for problem in env.problems],
        problem_types=[str(problem.problem_type) for problem in env.problems],
        visit_order=[idx + 1 for idx in env.state.visit_order],
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
    realized_rollouts: int = 100,
    obs_stats_path: str | None = None,
) -> dict[str, Any]:
    if policy_name not in HEURISTIC_MAP and rl_model is None:
        raise ValueError("For non-heuristic policy_name, provide rl_model.")

    is_dqn = rl_model is not None and rl_algorithm.lower() == "dqn"

    # Load obs normalizer from obs_stats.npz (only used for RL models).
    _fn = _load_obs_normalizer(obs_stats_path) if rl_model is not None else None

    def _norm(o: np.ndarray) -> np.ndarray:
        if _fn is None:
            return o
        return _fn(np.asarray(o, dtype=np.float32))

    records: list[EpisodeRecord] = []
    solved_criteria = solved_criteria_from_config(config)
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

        base_env = (
            _build_env(config=config, for_dqn=is_dqn, seed=seed + ep)
            if rl_model is not None
            else ExamStrategyEnv(config=config, random_seed=seed + ep)
        )
        obs, _ = base_env.reset(seed=seed + ep, options=reset_options)
        obs = _norm(obs)

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
            obs = _norm(obs)
            ep_reward += float(reward)

            state = base_env.state
            assert state is not None
            score_timeline.append(float(state.total_score))
            used_time_timeline.append(float(base_env.total_time_sec - state.remaining_time_sec))

        record = _episode_metrics(base_env, ep_reward, ep, ep_student_level, solved_criteria)
        record.score_timeline = score_timeline
        record.used_time_timeline = used_time_timeline

        if realized_rollouts > 0:
            rollout_rng = np.random.default_rng(seed + ep + 100000)
            r_mean, r_std = realized_score_rollout(
                state=base_env.state,
                problems=base_env.problems,
                n_rollouts=realized_rollouts,
                rng=rollout_rng,
            )
            record.realized_score_mean = r_mean
            record.realized_score_std = r_std

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
        "problem_avg_time_by_pid": summaries["problem_avg_time_by_pid"],
        "episode_records": [r.to_dict() for r in records],
    }


def _build_summary(records: list[EpisodeRecord], policy_name: str) -> dict[str, Any]:
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    overall = {
        "policy_name": policy_name,
        "mean_score": _mean([r.total_score for r in records]),
        "mean_realized_score": _mean([r.realized_score_mean for r in records]),
        "mean_realized_score_std": _mean([r.realized_score_std for r in records]),
        "mean_reward": _mean([r.total_reward for r in records]),
        "mean_solved_count": _mean([float(r.solved_count) for r in records]),
        "mean_visited_count": _mean([float(r.visited_count) for r in records]),
        "mean_coverage_fraction": _mean([r.coverage_fraction for r in records]),
        "mean_objective_dominance_rate": _mean([r.objective_dominance_rate for r in records]),
        "mean_subjective_confidence": _mean([r.mean_subjective_confidence for r in records]),
        "mean_subjective_solved_rate": _mean([r.subjective_solved_rate for r in records]),
        "mean_objective_solved_rate": _mean([r.objective_solved_rate for r in records]),
        "mean_top1_time_share": _mean([r.top1_time_share for r in records]),
        "mean_top2_time_share": _mean([r.top2_time_share for r in records]),
        "mean_remaining_time_sec": _mean([r.remaining_time_sec for r in records]),
        "mean_steps": _mean([float(r.steps) for r in records]),
    }

    by_level: dict[str, dict[str, float]] = {}
    for level in sorted({r.student_level for r in records}):
        chunk = [r for r in records if r.student_level == level]
        by_level[level] = {
            "mean_score": _mean([r.total_score for r in chunk]),
            "mean_realized_score": _mean([r.realized_score_mean for r in chunk]),
            "mean_realized_score_std": _mean([r.realized_score_std for r in chunk]),
            "mean_reward": _mean([r.total_reward for r in chunk]),
            "mean_solved_count": _mean([float(r.solved_count) for r in chunk]),
            "mean_visited_count": _mean([float(r.visited_count) for r in chunk]),
            "mean_coverage_fraction": _mean([r.coverage_fraction for r in chunk]),
            "mean_objective_dominance_rate": _mean([r.objective_dominance_rate for r in chunk]),
            "mean_subjective_confidence": _mean([r.mean_subjective_confidence for r in chunk]),
            "mean_subjective_solved_rate": _mean([r.subjective_solved_rate for r in chunk]),
            "mean_objective_solved_rate": _mean([r.objective_solved_rate for r in chunk]),
            "mean_top1_time_share": _mean([r.top1_time_share for r in chunk]),
            "mean_top2_time_share": _mean([r.top2_time_share for r in chunk]),
            "mean_remaining_time_sec": _mean([r.remaining_time_sec for r in chunk]),
            "mean_steps": _mean([float(r.steps) for r in chunk]),
        }

    num_problems = len(records[0].problem_time_spent) if records else 0
    problem_avg_time = []
    for i in range(num_problems):
        problem_avg_time.append(_mean([r.problem_time_spent[i] for r in records]))

    # When problem order is shuffled per episode, slot-based averages above
    # become hard to interpret. Aggregate by original pid as the stable key.
    pid_times: dict[int, list[float]] = {}
    pid_meta: dict[int, dict[str, Any]] = {}
    for record in records:
        for idx, pid in enumerate(record.problem_pids):
            pid_int = int(pid)
            pid_times.setdefault(pid_int, []).append(float(record.problem_time_spent[idx]))
            pid_meta.setdefault(
                pid_int,
                {
                    "pid": pid_int,
                    "difficulty_level": record.problem_difficulty_levels[idx],
                    "score": record.problem_scores[idx],
                    "problem_type": record.problem_types[idx],
                },
            )
    problem_avg_time_by_pid = []
    for pid in sorted(pid_times):
        item = dict(pid_meta[pid])
        item["avg_time_sec"] = _mean(pid_times[pid])
        problem_avg_time_by_pid.append(item)

    return {
        "overall": overall,
        "by_level": by_level,
        "problem_avg_time": problem_avg_time,
        "problem_avg_time_by_pid": problem_avg_time_by_pid,
    }


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
