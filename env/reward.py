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
    was_correct: bool | None,
    reward_cfg: dict,
) -> float:
    reward = 0.0

    if was_correct is True and problem is not None:
        reward += _rw(reward_cfg, "correct_answer", 1.0) * float(problem.score)
    elif was_correct is False:
        reward += _rw(reward_cfg, "wrong_answer", -0.25)

    if action_name == "skip":
        reward += _rw(reward_cfg, "skip_penalty", -0.02)

    time_spent = max(0.0, prev_state.remaining_time_sec - next_state.remaining_time_sec)
    reward += _rw(reward_cfg, "time_penalty_per_sec", -0.001) * time_spent

    if problem is not None and problem_idx is not None:
        idx = problem_idx
        curr = next_state.progress[idx]
        if curr.time_spent_sec > problem.avg_time * 1.8 and curr.status != ProblemStatus.SOLVED:
            reward += _rw(reward_cfg, "overfocus_penalty", -0.2)

        if curr.status in {ProblemStatus.FAILED, ProblemStatus.GIVEN_UP} and problem.difficulty <= 0.35:
            reward += _rw(reward_cfg, "easy_miss_penalty", -0.25)

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
        if progress.status == ProblemStatus.NOT_VISITED and problems[i].difficulty <= 0.35:
            reward += _rw(reward_cfg, "easy_unvisited_penalty", -0.2)

    return float(reward)
