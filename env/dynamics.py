from __future__ import annotations

import math

from .problem import Problem
from .state import ExamState, ProblemStatus
from .student import StudentProfile


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def guessing_prob(problem: Problem) -> float:
    if problem.problem_type == "objective" and problem.choice_rate:
        return 1.0 / max(len(problem.choice_rate), 1)
    return 0.02


def confidence_params(
    problem: Problem,
    student: StudentProfile,
    time_pressure: float = 0.0,
) -> tuple[float, float, float]:
    c_m = guessing_prob(problem)
    efficiency = 0.45 * student.skill_global + 0.30 * student.skill_accuracy + 0.25 * student.skill_speed
    hardness = _clamp(0.55 * problem.difficulty + 0.45 * problem.error_rate)
    stress_load = _clamp(time_pressure) * (1.0 - _clamp(student.stress_tolerance))

    # Confidence growth depends on problem hardness and student ability,
    # not on a pre-labeled per-problem average solve time.
    lambda_m = (0.004 + 0.012 * efficiency) * (0.45 + 0.55 * (1.0 - hardness))
    tau_m = 15.0 + 90.0 * hardness * (1.10 - 0.35 * student.skill_speed)

    # Lower stress tolerance hurts confidence growth more as time pressure rises.
    lambda_m *= 1.0 - 0.35 * stress_load
    tau_m *= 1.0 + 0.60 * stress_load
    return _clamp(c_m), max(lambda_m, 1e-4), max(tau_m, 0.0)


def confidence_curve(
    problem: Problem,
    student: StudentProfile,
    time_spent: float,
    time_pressure: float = 0.0,
) -> float:
    c_m, lambda_m, tau_m = confidence_params(problem, student, time_pressure=time_pressure)
    effective_time = max(time_spent - tau_m, 0.0)
    confidence = c_m + (1.0 - c_m) * (1.0 - math.exp(-lambda_m * effective_time))
    return _clamp(confidence)


def expected_total_score(state: ExamState, problems: list[Problem]) -> float:
    return float(
        sum(float(problem.score) * progress.confidence_score for problem, progress in zip(problems, state.progress))
    )


def apply_time_cost(state: ExamState, delta_time_sec: float) -> float:
    if state.remaining_time_sec <= 0.0:
        return 0.0
    spent = min(float(delta_time_sec), state.remaining_time_sec)
    state.remaining_time_sec -= spent
    return spent


def solve_more(
    state: ExamState,
    problem_idx: int,
    delta_time_sec: float,
    problem: Problem,
    student: StudentProfile,
    total_time_sec: float,
) -> float:
    spent = apply_time_cost(state, delta_time_sec)
    if spent <= 0.0:
        return 0.0
    progress = state.progress[problem_idx]
    progress.status = ProblemStatus.IN_PROGRESS
    progress.time_spent_sec += spent
    time_pressure = 1.0 - (state.remaining_time_sec / max(float(total_time_sec), 1.0))
    progress.confidence_score = confidence_curve(
        problem=problem,
        student=student,
        time_spent=progress.time_spent_sec,
        time_pressure=time_pressure,
    )
    return spent


def move_next(state: ExamState, problem_idx: int, target_problem_idx: int) -> None:
    state.progress[problem_idx].status = ProblemStatus.MOVED_ON
    state.current_problem_idx = target_problem_idx
    if state.progress[target_problem_idx].status == ProblemStatus.NOT_VISITED:
        state.progress[target_problem_idx].status = ProblemStatus.IN_PROGRESS
        state.visit_order.append(target_problem_idx)
    else:
        state.progress[target_problem_idx].status = ProblemStatus.IN_PROGRESS
