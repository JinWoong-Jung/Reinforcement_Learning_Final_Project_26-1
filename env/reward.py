from __future__ import annotations

from .problem import Problem
from .state import ExamState


def _rw(cfg: dict, key: str, default: float) -> float:
    return float(cfg.get(key, default))


def expected_utility(state: ExamState, problems: list[Problem]) -> float:
    return float(sum(float(problem.score) * progress.confidence_score for problem, progress in zip(problems, state.progress)))


def compute_step_reward(
    prev_state: ExamState,
    next_state: ExamState,
    problems: list[Problem],
    action_name: str,
    reward_cfg: dict,
) -> float:
    prev_u = expected_utility(prev_state, problems)
    next_u = expected_utility(next_state, problems)
    reward = next_u - prev_u

    if action_name == "next":
        reward += _rw(reward_cfg, "next_penalty", 0.0)

    return float(reward)


def compute_terminal_reward(
    state: ExamState,
    problems: list[Problem],
    reward_cfg: dict,
) -> float:
    reward = 0.0
    if state.remaining_time_sec <= 0:
        reward += _rw(reward_cfg, "timeout_penalty", 0.0)
    if state.is_all_terminal():
        reward += _rw(reward_cfg, "completion_bonus", 0.0)
    return float(reward)
