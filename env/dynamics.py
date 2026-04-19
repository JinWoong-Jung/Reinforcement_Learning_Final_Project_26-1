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

    - objective: fixed 0.2
    - subjective: configurable epsilon (default 0.0)
    """
    if problem.problem_type == "objective":
        return 0.2
    subjective_floor = _dcfg(dynamics_cfg, "subjective_floor", default=0.0)
    return float(subjective_floor)


def _difficulty_anchor(problem: Problem, dynamics_cfg: dict | None = None) -> float:
    """Return the main difficulty anchor in [0, 1].

    Priority:
      1. difficulty
      2. correct_rate-derived difficulty as a legacy fallback
    Controlled by dynamics_cfg['anchor_source'] = 'difficulty' | 'correct_rate'.
    """
    anchor_source = _scfg(dynamics_cfg, "anchor_source", default="difficulty")
    if anchor_source == "difficulty":
        return _clamp(problem.difficulty)
    if anchor_source == "correct_rate" and problem.correct_rate is not None:
        return _clamp(1.0 - problem.correct_rate)
    return _clamp(problem.difficulty)


def _student_theta(student: StudentProfile) -> float:
    """Student ability is controlled directly by the profile theta."""
    return float(student.theta)


def confidence_params(
    problem: Problem,
    student: StudentProfile,
    dynamics_cfg: dict | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return interpretable parameters for the logistic confidence model.

    Returns:
        (floor, theta, beta, gamma, alpha, tau)

        floor  – chance-level probability floor (c_i)
        theta  – student ability logit from the student profile
        beta   – difficulty weight applied to difficulty_anchor
        gamma  – ambiguity weight applied to choice_entropy
        alpha  – time learning rate (slope of log-time curve)
        tau    – time scale in seconds
    """
    floor = guessing_prob(problem, dynamics_cfg)

    theta = _student_theta(student)

    beta = _dcfg(dynamics_cfg, "beta", default=2.9)
    gamma = _dcfg(dynamics_cfg, "ambiguity_weight", default=1.7)
    alpha = _dcfg(dynamics_cfg, "alpha", default=1.6)
    tau = max(_dcfg(dynamics_cfg, "tau", default=200.0), 1.0)

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
        d      = difficulty anchor in [0,1]  (difficulty by default)
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


def confidence_static_params(
    problem: Problem,
    student: StudentProfile,
    dynamics_cfg: dict | None = None,
) -> tuple[float, float, float, float]:
    """Return time-independent parameters for fast marginal-gain computation.

    Returns:
        (floor, static_logit, alpha, tau)

        static_logit = theta - beta*diff_anchor - gamma*ambiguity

    The full confidence at time t is:
        p = floor + (1-floor) * sigmoid(static_logit + alpha * log(1 + t/tau))

    Cache this per-problem per-episode (never changes within an episode).
    """
    floor, theta, beta, gamma, alpha, tau = confidence_params(problem, student, dynamics_cfg)
    diff_anchor = _difficulty_anchor(problem, dynamics_cfg)
    ambiguity = choice_entropy(problem)
    static_logit = theta - beta * diff_anchor - gamma * ambiguity
    return float(floor), float(static_logit), float(alpha), float(tau)


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
