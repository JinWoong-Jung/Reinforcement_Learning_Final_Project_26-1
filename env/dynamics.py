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
    if progress.status in {ProblemStatus.SUBMITTED, ProblemStatus.GIVEN_UP}:
        progress.status = ProblemStatus.IN_PROGRESS
    progress.time_spent_sec += spent
    if progress.status == ProblemStatus.NOT_VISITED:
        progress.status = ProblemStatus.IN_PROGRESS
    state.remaining_time_sec -= spent
    return spent


def _apply_hidden_result(state: ExamState, problem_idx: int, problem: Problem, is_correct: bool) -> None:
    progress = state.progress[problem_idx]
    prev_correct = progress.judged_correct
    if prev_correct is True:
        state.total_score -= float(problem.score)

    progress.judged_correct = bool(is_correct)
    if is_correct:
        state.total_score += float(problem.score)


def submission_confidence(
    problem: Problem,
    time_spent: float,
    estimated_prob: float,
    submit_count: int,
) -> float:
    time_ratio = time_spent / max(problem.avg_time, 1.0)
    time_component = 0.25 + 0.45 * (1.0 - math.exp(-time_ratio))
    prob_component = 0.30 * estimated_prob
    retry_penalty = 0.08 * max(submit_count - 1, 0)
    confidence = time_component + prob_component - retry_penalty
    return _clamp(confidence)


def submit_answer(
    state: ExamState,
    problem_idx: int,
    problem: Problem,
    student: StudentProfile,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    progress = state.progress[problem_idx]
    if progress.status not in {ProblemStatus.IN_PROGRESS}:
        return False, 0.0

    prob = correct_prob(
        problem=problem,
        student=student,
        time_spent=progress.time_spent_sec,
        remaining_time_sec=state.remaining_time_sec,
    )
    is_correct = bool(rng.random() < prob)
    progress.submit_count += 1
    progress.status = ProblemStatus.SUBMITTED
    progress.confidence_score = submission_confidence(
        problem=problem,
        time_spent=progress.time_spent_sec,
        estimated_prob=prob,
        submit_count=progress.submit_count,
    )
    _apply_hidden_result(state=state, problem_idx=problem_idx, problem=problem, is_correct=is_correct)

    return is_correct, prob


def guess_answer(
    state: ExamState,
    problem_idx: int,
    problem: Problem,
    student: StudentProfile,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    progress = state.progress[problem_idx]
    if progress.status not in {ProblemStatus.IN_PROGRESS}:
        return False, 0.0

    if problem.problem_type == "objective" and problem.choice_rate:
        choice_count = max(len(problem.choice_rate), 1)
        uniform_prob = 1.0 / choice_count
        correct_key = str(problem.actual_answer) if problem.actual_answer is not None else None
        empirical_prob = float(problem.choice_rate.get(correct_key, uniform_prob))
        prob = (1.0 - student.skill_guess) * uniform_prob + student.skill_guess * empirical_prob
    else:
        base_prob = 0.01 + 0.08 * student.skill_guess
        prob = base_prob * (1.0 - 0.6 * problem.difficulty)

    prob = _clamp(prob)
    is_correct = bool(rng.random() < prob)
    progress.submit_count += 1
    progress.status = ProblemStatus.GIVEN_UP
    progress.confidence_score = _clamp(0.15 + 0.35 * prob)
    _apply_hidden_result(state=state, problem_idx=problem_idx, problem=problem, is_correct=is_correct)

    return is_correct, prob
