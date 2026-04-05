from __future__ import annotations

from .problem import Problem
from .state import ExamState, ProblemStatus


def _rw(cfg: dict, key: str, default: float) -> float:
    return float(cfg.get(key, default))


def compute_step_reward(
    prev_state: ExamState,
    next_state: ExamState,
    problem_idx: int | None,
    problem: Problem | None,
    action_name: str,
    reward_cfg: dict,
) -> float:
    reward = 0.0

    if action_name == "solve_more":
        reward += _rw(reward_cfg, "solve_more_penalty", -0.02)
    elif action_name == "submit":
        reward += _rw(reward_cfg, "submit_bonus", 0.1)
    elif action_name == "give_up":
        reward += _rw(reward_cfg, "give_up_penalty", -0.05)
    elif action_name == "skip":
        reward += _rw(reward_cfg, "skip_penalty", -0.02)
    elif action_name == "review":
        reward += _rw(reward_cfg, "review_bonus", 0.0)

    time_spent = max(0.0, prev_state.remaining_time_sec - next_state.remaining_time_sec)
    reward += _rw(reward_cfg, "time_penalty_per_sec", -0.001) * time_spent

    if problem is not None and problem_idx is not None:
        idx = problem_idx
        curr = next_state.progress[idx]
        if action_name == "solve_more" and curr.time_spent_sec > problem.avg_time * 1.8:
            reward += _rw(reward_cfg, "overfocus_penalty", -0.2)

    return float(reward)


def compute_terminal_reward(
    state: ExamState,
    problems: list[Problem],
    reward_cfg: dict,
) -> float:
    timeout_penalty = _rw(reward_cfg, "timeout_penalty", -1.0)
    remaining_time_bonus = _rw(reward_cfg, "remaining_time_bonus", 0.001)

    reward = 0.0
    if state.remaining_time_sec <= 0:
        reward += timeout_penalty

    reward += remaining_time_bonus * state.remaining_time_sec

    for i, progress in enumerate(state.progress):
        if progress.judged_correct is True:
            reward += _rw(reward_cfg, "correct_answer", 1.0) * float(problems[i].score)
        elif progress.judged_correct is False and progress.status == ProblemStatus.SUBMITTED:
            reward += _rw(reward_cfg, "wrong_answer", -0.25)

        if progress.status == ProblemStatus.GIVEN_UP and problems[i].difficulty <= 0.35:
            reward += _rw(reward_cfg, "easy_miss_penalty", -0.25)

        if progress.status == ProblemStatus.NOT_VISITED and problems[i].difficulty <= 0.35:
            reward += _rw(reward_cfg, "easy_unvisited_penalty", -0.2)

    return float(reward)
