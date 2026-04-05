from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from env.exam_env import ExamStrategyEnv
from env.problem import Problem


HeuristicFn = Callable[[ExamStrategyEnv], int]


@dataclass
class EpisodeStats:
    total_reward: float
    total_score: float
    solved_count: int
    remaining_time_sec: float
    steps: int
    done: bool


def target_time_budget(problem: Problem, policy_name: str) -> float:
    if policy_name == "index_order":
        return problem.avg_time * 0.95
    if policy_name == "easy_first":
        return problem.avg_time * (0.75 if problem.score <= 3 else 0.60)
    if policy_name == "high_score_first":
        return problem.avg_time * (1.15 if problem.score >= 4 else 0.70)
    if policy_name == "score_time_ratio":
        return problem.avg_time * (1.00 if problem.score >= 4 else 0.65)
    return problem.avg_time


def policy_index_order(env: ExamStrategyEnv) -> int:
    return 0


def policy_easy_first(env: ExamStrategyEnv) -> int:
    assert env.state is not None
    problem = env.problems[env.state.current_problem_idx]
    budget = target_time_budget(problem, "easy_first")
    return 0 if env.state.progress[env.state.current_problem_idx].time_spent_sec < budget else 1


def policy_high_score_first(env: ExamStrategyEnv) -> int:
    assert env.state is not None
    problem = env.problems[env.state.current_problem_idx]
    budget = target_time_budget(problem, "high_score_first")
    return 0 if env.state.progress[env.state.current_problem_idx].time_spent_sec < budget else 1


def policy_expected_score_time_ratio(env: ExamStrategyEnv) -> int:
    assert env.state is not None
    problem = env.problems[env.state.current_problem_idx]
    budget = target_time_budget(problem, "score_time_ratio")
    return 0 if env.state.progress[env.state.current_problem_idx].time_spent_sec < budget else 1


HEURISTIC_POLICIES: dict[str, HeuristicFn] = {
    "index_order": policy_index_order,
    "easy_first": policy_easy_first,
    "high_score_first": policy_high_score_first,
    "score_time_ratio": policy_expected_score_time_ratio,
}


def heuristic_action(env: ExamStrategyEnv, policy_name: str) -> int:
    selector = HEURISTIC_POLICIES.get(policy_name)
    if selector is None:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")

    if policy_name == "index_order":
        assert env.state is not None
        problem = env.problems[env.state.current_problem_idx]
        budget = target_time_budget(problem, "index_order")
        return 0 if env.state.progress[env.state.current_problem_idx].time_spent_sec < budget else 1
    return selector(env)


def run_heuristic_episode(
    env: ExamStrategyEnv,
    policy_name: str,
    max_steps: int = 10000,
    reset_seed: int | None = None,
) -> EpisodeStats:
    _, _ = env.reset(seed=reset_seed)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = heuristic_action(env, policy_name)
        _, r, terminated, truncated, _ = env.step(action)
        total_reward += float(r)
        done = bool(terminated or truncated)
        steps += 1

    assert env.state is not None
    return EpisodeStats(
        total_reward=float(total_reward),
        total_score=float(env.state.total_score),
        solved_count=int(env.state.solved_count()),
        remaining_time_sec=float(env.state.remaining_time_sec),
        steps=int(steps),
        done=bool(done),
    )


def evaluate_heuristic_policy(
    env_factory: Callable[[], ExamStrategyEnv],
    policy_name: str,
    episodes: int = 50,
    seed: int = 42,
) -> dict:
    metrics = {
        "total_reward": [],
        "total_score": [],
        "solved_count": [],
        "remaining_time_sec": [],
        "steps": [],
    }
    for ep in range(episodes):
        env = env_factory()
        stats = run_heuristic_episode(env=env, policy_name=policy_name, reset_seed=seed + ep)
        metrics["total_reward"].append(stats.total_reward)
        metrics["total_score"].append(stats.total_score)
        metrics["solved_count"].append(stats.solved_count)
        metrics["remaining_time_sec"].append(stats.remaining_time_sec)
        metrics["steps"].append(stats.steps)

    return {
        "policy": policy_name,
        "episodes": episodes,
        "mean_total_reward": float(np.mean(metrics["total_reward"])),
        "mean_total_score": float(np.mean(metrics["total_score"])),
        "mean_solved_count": float(np.mean(metrics["solved_count"])),
        "mean_remaining_time_sec": float(np.mean(metrics["remaining_time_sec"])),
        "mean_steps": float(np.mean(metrics["steps"])),
    }


def evaluate_all_heuristics(
    env_factory: Callable[[], ExamStrategyEnv],
    episodes: int = 50,
    seed: int = 42,
) -> list[dict]:
    return [
        evaluate_heuristic_policy(env_factory=env_factory, policy_name=name, episodes=episodes, seed=seed)
        for name in HEURISTIC_POLICIES
    ]
