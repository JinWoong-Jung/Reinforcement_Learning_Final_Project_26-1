from __future__ import annotations

import math

from .problem import Problem, choice_entropy
from .state import ExamState, ProblemStatus
from .student import StudentProfile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _dcfg(cfg: dict | None, *path: str, default: float) -> float:
    """Read a float from a nested config dict."""
    current = cfg or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _scfg(cfg: dict | None, *path: str, default: str) -> str:
    """Read a str from a nested config dict."""
    current = cfg or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return str(current)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def guessing_prob(problem: Problem, dynamics_cfg: dict | None = None) -> float:
    """Chance-level floor probability for a problem.

    - objective: 1 / number_of_choices  (e.g. 0.2 for 5-choice)
    - subjective: configurable epsilon (default 0.02)
    """
    if problem.problem_type == "objective" and problem.choice_rate:
        return 1.0 / max(len(problem.choice_rate), 1)
    subjective_floor = _dcfg(dynamics_cfg, "subjective_floor", default=0.02)
    return float(subjective_floor)


def _difficulty_anchor(problem: Problem, dynamics_cfg: dict | None = None) -> float:
    """Return the main difficulty anchor in [0, 1].

    Priority:
      1. correct_rate  (1 - correct_rate gives difficulty)
      2. difficulty field as fallback
    Controlled by dynamics_cfg['anchor_source'] = 'correct_rate' | 'difficulty'.
    """
    anchor_source = _scfg(dynamics_cfg, "anchor_source", default="correct_rate")
    if anchor_source == "correct_rate" and problem.correct_rate is not None:
        return _clamp(1.0 - problem.correct_rate)
    return _clamp(problem.difficulty)


def _student_ability(student: StudentProfile, dynamics_cfg: dict | None = None) -> float:
    """Weighted combination of student skill dimensions, in [0, 1]."""
    w_global = _dcfg(dynamics_cfg, "ability_weights", "skill_global", default=0.45)
    w_accuracy = _dcfg(dynamics_cfg, "ability_weights", "skill_accuracy", default=0.30)
    w_speed = _dcfg(dynamics_cfg, "ability_weights", "skill_speed", default=0.25)
    total_w = w_global + w_accuracy + w_speed
    if total_w <= 0.0:
        total_w = 1.0
    ability = (
        w_global * student.skill_global
        + w_accuracy * student.skill_accuracy
        + w_speed * student.skill_speed
    ) / total_w
    return _clamp(ability)


def confidence_params(
    problem: Problem,
    student: StudentProfile,
    dynamics_cfg: dict | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return interpretable parameters for the logistic confidence model.

    Returns:
        (floor, theta, beta, gamma, alpha, tau)

        floor  – chance-level probability floor (c_i)
        theta  – student ability logit: theta_scale * (ability - 0.5)
        beta   – difficulty weight applied to difficulty_anchor
        gamma  – ambiguity weight applied to choice_entropy
        alpha  – time learning rate (slope of log-time curve)
        tau    – time scale in seconds
    """
    floor = guessing_prob(problem, dynamics_cfg)

    ability = _student_ability(student, dynamics_cfg)
    theta_scale = _dcfg(dynamics_cfg, "theta_scale", default=3.0)
    theta = theta_scale * (ability - 0.5)

    beta = _dcfg(dynamics_cfg, "beta", default=2.0)
    gamma = _dcfg(dynamics_cfg, "ambiguity_weight", default=0.5)
    alpha = _dcfg(dynamics_cfg, "alpha", default=2.0)
    tau = max(_dcfg(dynamics_cfg, "tau", default=60.0), 1.0)

    return float(floor), float(theta), float(beta), float(gamma), float(alpha), float(tau)


def confidence_curve(
    problem: Problem,
    student: StudentProfile,
    time_spent: float,
    dynamics_cfg: dict | None = None,
) -> float:
    """Compute p_i(t | s) using an interpretable logistic model.

    Formula:
        p = floor + (1 - floor) * sigmoid(theta - beta*d - gamma*a + alpha*log(1 + t/tau))

    Where:
        floor  = chance-level guessing probability
        theta  = student ability logit  (positive → more able)
        d      = difficulty anchor in [0,1]  (1 - correct_rate or difficulty)
        a      = ambiguity feature in [0,1]  (normalized choice entropy)
        alpha  = time learning rate
        tau    = time scale (seconds)

    Properties guaranteed:
        - Monotonically increasing in time_spent.
        - Harder problems (higher d) → lower confidence at same time.
        - More ambiguous problems (higher a) → lower confidence at same time.
        - More able students (higher theta) → higher confidence at same time.
        - floor ≤ p ≤ 1.
    """
    floor, theta, beta, gamma, alpha, tau = confidence_params(problem, student, dynamics_cfg)

    diff_anchor = _difficulty_anchor(problem, dynamics_cfg)
    ambiguity = choice_entropy(problem)

    time_logit = alpha * math.log(1.0 + max(time_spent, 0.0) / tau)
    logit = theta - beta * diff_anchor - gamma * ambiguity + time_logit

    p = floor + (1.0 - floor) * _sigmoid(logit)
    return _clamp(p)


def expected_total_score(state: ExamState, problems: list[Problem]) -> float:
    return float(
        sum(float(problem.score) * progress.effective_confidence(problem) for problem, progress in zip(problems, state.progress))
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
    dynamics_cfg: dict | None = None,
) -> float:
    """Apply one solve_more action: consume time and update confidence.

    time_pressure is retained internally but no longer affects the
    confidence model (it was a hidden stressor that made the model less
    interpretable).  The logistic model directly captures ability and
    difficulty without needing a time-pressure multiplier.
    """
    spent = apply_time_cost(state, delta_time_sec)
    if spent <= 0.0:
        return 0.0
    progress = state.progress[problem_idx]
    progress.status = ProblemStatus.IN_PROGRESS
    progress.time_spent_sec += spent

    scalar_confidence = confidence_curve(
        problem=problem,
        student=student,
        time_spent=progress.time_spent_sec,
        dynamics_cfg=dynamics_cfg,
    )
    if problem.problem_type == "objective":
        scalar_confidence = max(progress.effective_confidence(problem), scalar_confidence)
    progress.sync_from_scalar(problem, scalar_confidence)
    return spent


def move_next(state: ExamState, problem_idx: int, target_problem_idx: int) -> None:
    state.progress[problem_idx].status = ProblemStatus.MOVED_ON
    state.current_problem_idx = target_problem_idx
    if state.progress[target_problem_idx].status == ProblemStatus.NOT_VISITED:
        state.progress[target_problem_idx].status = ProblemStatus.IN_PROGRESS
        state.visit_order.append(target_problem_idx)
    else:
        state.progress[target_problem_idx].status = ProblemStatus.IN_PROGRESS
