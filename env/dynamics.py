from __future__ import annotations

import math

import numpy as np

from .problem import Problem
from .state import ExamState, ProblemStatus
from .student import StudentProfile


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _to_odds(p: float) -> float:
    p = _clamp(p, 1e-6, 1.0 - 1e-6)
    return p / (1.0 - p)


def _from_odds(odds: float) -> float:
    return odds / (1.0 + odds)


def correct_prob(
    problem: Problem,
    student: StudentProfile,
    time_spent: float,
    remaining_time_sec: float | None = None,
) -> float:
    topic_skill = student.topic_level(problem.topic)
    skill_mix = (
        0.35 * student.skill_global
        + 0.25 * topic_skill
        + 0.25 * student.skill_accuracy
        + 0.15 * student.skill_speed
    )
    time_scale = max(15.0, problem.avg_time / max(0.35, student.skill_speed))
    time_effect = 1.0 - math.exp(-time_spent / time_scale)

    hardness = _clamp(0.5 * problem.difficulty + 0.5 * problem.error_rate)
    base = 0.08 + 0.50 * skill_mix + 0.24 * (1.0 - hardness)
    prob = base * (0.40 + 0.80 * time_effect)

    # Exponential skill lift on odds:
    # Higher-skill students gain disproportionate benefit, especially on harder items.
    skill_centered = (skill_mix - 0.5) * 2.0  # roughly in [-1, 1]
    skill_exponent = 1.8 * skill_centered * (0.55 + hardness)
    odds = _to_odds(prob) * math.exp(skill_exponent)
    prob = _from_odds(odds)

    # Easy-item uplift: strong students should almost always convert easy problems.
    easy_factor = (1.0 - hardness)
    easy_uplift = 0.20 * easy_factor * (0.25 + 0.75 * skill_mix)
    prob += easy_uplift

    if time_spent <= 0.0:
        prob = max(prob, 0.18 * student.skill_guess)

    if remaining_time_sec is not None:
        stress_window = max(problem.avg_time, 1.0)
        if remaining_time_sec < stress_window:
            stress_penalty = (1.0 - student.stress_tolerance) * 0.12
            prob -= stress_penalty

    return _clamp(prob)


def solve_more(
    state: ExamState,
    problem_idx: int,
    delta_time_sec: float,
) -> float:
    if state.remaining_time_sec <= 0.0:
        return 0.0

    spent = min(float(delta_time_sec), state.remaining_time_sec)
    progress = state.progress[problem_idx]
    if progress.status in {ProblemStatus.SOLVED, ProblemStatus.FAILED, ProblemStatus.GIVEN_UP}:
        return 0.0

    progress.time_spent_sec += spent
    if progress.status == ProblemStatus.NOT_VISITED:
        progress.status = ProblemStatus.IN_PROGRESS
    state.remaining_time_sec -= spent
    return spent


def submit_answer(
    state: ExamState,
    problem_idx: int,
    problem: Problem,
    student: StudentProfile,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    progress = state.progress[problem_idx]
    if progress.status in {ProblemStatus.SOLVED, ProblemStatus.GIVEN_UP}:
        return False, 0.0

    prob = correct_prob(
        problem=problem,
        student=student,
        time_spent=progress.time_spent_sec,
        remaining_time_sec=state.remaining_time_sec,
    )
    is_correct = bool(rng.random() < prob)
    progress.submit_count += 1

    if is_correct:
        progress.status = ProblemStatus.SOLVED
        state.total_score += float(problem.score)
    else:
        progress.status = ProblemStatus.FAILED

    return is_correct, prob


def give_up_problem(state: ExamState, problem_idx: int) -> None:
    progress = state.progress[problem_idx]
    if progress.status in {ProblemStatus.SOLVED, ProblemStatus.FAILED, ProblemStatus.GIVEN_UP}:
        return
    progress.status = ProblemStatus.GIVEN_UP
