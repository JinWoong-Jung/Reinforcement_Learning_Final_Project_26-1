from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from env.exam_env import ExamStrategyEnv
from env.problem import Problem
from env.state import ProblemStatus


HeuristicFn = Callable[[ExamStrategyEnv], int]


@dataclass
class EpisodeStats:
    total_reward: float
    total_score: float
    solved_count: int
    remaining_time_sec: float
    steps: int
    done: bool


def _available_problem_indices(env: ExamStrategyEnv) -> list[int]:
    assert env.state is not None
    idxs: list[int] = []
    threshold = float(getattr(env, "review_conf_threshold", 0.7))
    for i, p in enumerate(env.state.progress):
        if p.status == ProblemStatus.SUBMITTED and p.confidence_score >= threshold:
            continue
        if p.status == ProblemStatus.GIVEN_UP and p.confidence_score >= threshold:
            continue
        if p.status != ProblemStatus.SUBMITTED or p.confidence_score < threshold:
            idxs.append(i)
    return idxs


def _priority_expected_score_per_time(env: ExamStrategyEnv, idx: int) -> float:
    assert env.state is not None
    problem = env.problems[idx]
    current_spent = env.state.progress[idx].time_spent_sec
    projected_time = current_spent + env.action_time_unit_sec
    submit_penalty = 1.0 + 0.5 * env.state.progress[idx].submit_count
    denom = max(projected_time * submit_penalty, 1.0)
    return float(problem.score) / denom


def policy_index_order(env: ExamStrategyEnv) -> int:
    candidates = _available_problem_indices(env)
    return min(candidates) if candidates else 0


def policy_easy_first(env: ExamStrategyEnv) -> int:
    candidates = _available_problem_indices(env)
    if not candidates:
        return 0
    # Agent does not observe true difficulty; use low-score-first as a public proxy.
    return min(candidates, key=lambda i: (env.problems[i].score, i))


def policy_high_score_first(env: ExamStrategyEnv) -> int:
    candidates = _available_problem_indices(env)
    if not candidates:
        return 0
    return max(candidates, key=lambda i: (env.problems[i].score, -i))


def policy_expected_score_time_ratio(env: ExamStrategyEnv) -> int:
    candidates = _available_problem_indices(env)
    if not candidates:
        return 0
    return max(candidates, key=lambda i: _priority_expected_score_per_time(env, i))


def target_time_budget(problem: Problem, policy_name: str) -> float:
    if policy_name == "index_order":
        return problem.avg_time * 0.95
    if policy_name == "easy_first":
        return problem.avg_time * (0.80 if problem.score <= 3 else 0.70)
    if policy_name == "high_score_first":
        return problem.avg_time * (1.2 if problem.score >= 4 else 0.8)
    if policy_name == "score_time_ratio":
        return problem.avg_time * (1.10 if problem.score >= 4 else 0.75)
    return problem.avg_time


HEURISTIC_POLICIES: dict[str, HeuristicFn] = {
    "index_order": policy_index_order,
    "easy_first": policy_easy_first,
    "high_score_first": policy_high_score_first,
    "score_time_ratio": policy_expected_score_time_ratio,
}


def heuristic_action(env: ExamStrategyEnv, policy_name: str) -> np.ndarray:
    assert env.state is not None
    selector = HEURISTIC_POLICIES.get(policy_name)
    if selector is None:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")

    candidates = _available_problem_indices(env)
    if not candidates:
        return np.array([0, 3], dtype=np.int64)  # skip

    current_idx = int(env.state.current_problem_idx)
    idx = int(np.clip(selector(env), 0, env.num_problems - 1))
    current_progress = env.state.progress[current_idx]
    current_problem = env.problems[current_idx]

    if idx != current_idx:
        return np.array([idx, 3], dtype=np.int64)  # move to target problem

    progress = current_progress
    problem = current_problem
    threshold = float(getattr(env, "review_conf_threshold", 0.7))
    if progress.status in {ProblemStatus.SUBMITTED, ProblemStatus.GIVEN_UP} and progress.confidence_score < threshold and env.state.remaining_time_sec > 0:
        return np.array([current_idx, 0], dtype=np.int64)  # reopen by solving more
    budget = target_time_budget(problem, policy_name)
    if progress.time_spent_sec < budget and env.state.remaining_time_sec > 0:
        return np.array([current_idx, 0], dtype=np.int64)  # solve_more

    if progress.status == ProblemStatus.IN_PROGRESS and progress.time_spent_sec > 0:
        return np.array([current_idx, 1], dtype=np.int64)  # submit

    unseen = [i for i in range(env.num_problems) if env.state.progress[i].status == ProblemStatus.NOT_VISITED and i != current_idx]
    if unseen:
        return np.array([unseen[0], 3], dtype=np.int64)

    visited = [i for i in candidates if i != current_idx]
    if visited:
        return np.array([visited[0], 3], dtype=np.int64)

    return np.array([current_idx, 0], dtype=np.int64)


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
        assert env.state is not None
        candidates = _available_problem_indices(env)
        if not candidates:
            break

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
        stats = run_heuristic_episode(
            env=env,
            policy_name=policy_name,
            reset_seed=seed + ep,
        )
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
    setups = list(HEURISTIC_POLICIES.items())
    return [
        evaluate_heuristic_policy(
            env_factory=env_factory,
            policy_name=name,
            episodes=episodes,
            seed=seed,
        )
        for name, _ in setups
    ]
