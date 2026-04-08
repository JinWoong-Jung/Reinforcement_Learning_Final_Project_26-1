from __future__ import annotations

from .problem import Problem
from .state import ExamState, ProblemStatus


def _cfg(cfg: dict | None, *path: str, default):
    current = cfg or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _rw(cfg: dict | None, *path: str, default: float) -> float:
    return float(_cfg(cfg, *path, default=default))


def _iw(cfg: dict | None, *path: str, default: int) -> int:
    return int(_cfg(cfg, *path, default=default))


def expected_utility(state: ExamState, problems: list[Problem]) -> float:
    return float(sum(float(problem.score) * progress.confidence_score for problem, progress in zip(problems, state.progress)))


def _visited_count(state: ExamState) -> int:
    return len(set(state.visit_order))


def _coverage_bonus(prev_state: ExamState, problems: list[Problem], reward_cfg: dict) -> float:
    total = max(len(problems), 1)
    visited_before = _visited_count(prev_state)
    remaining_fraction = max(total - visited_before, 0) / float(total)
    fixed_bonus = _rw(reward_cfg, "next", "new_problem_bonus", default=0.0)
    bonus_power = _rw(reward_cfg, "next", "coverage_bonus_power", default=1.0)
    scaled_bonus = _rw(reward_cfg, "next", "coverage_bonus_scale", default=0.0) * (remaining_fraction ** bonus_power)
    return fixed_bonus + scaled_bonus


def _next_transition_shaping(prev_state: ExamState, problems: list[Problem], reward_cfg: dict) -> float:
    current_idx = prev_state.current_problem_idx
    problem = problems[current_idx]
    confidence = float(prev_state.progress[current_idx].confidence_score)
    level = str(problem.difficulty_level)
    exit_cfg = _cfg(reward_cfg, "next", "difficulty_exit", default={})

    if level in {"하", "중하"}:
        if confidence < _rw(exit_cfg, "easy", "low_conf_threshold", default=0.5):
            return _rw(exit_cfg, "easy", "low_conf_penalty", default=-0.08)
        if confidence >= _rw(exit_cfg, "easy", "ready_threshold", default=0.7):
            return _rw(exit_cfg, "easy", "ready_bonus", default=0.08)
        return 0.0

    if level in {"중", "중상"}:
        if confidence < _rw(exit_cfg, "mid", "low_conf_threshold", default=0.4):
            return _rw(exit_cfg, "mid", "low_conf_penalty", default=-0.04)
        if confidence >= _rw(exit_cfg, "mid", "ready_threshold", default=0.6):
            return _rw(exit_cfg, "mid", "ready_bonus", default=0.05)
        return 0.0

    if level in {"상", "최상"}:
        if confidence >= _rw(exit_cfg, "hard", "ready_threshold", default=0.35):
            return _rw(exit_cfg, "hard", "ready_bonus", default=0.06)
        return 0.0

    return 0.0


def compute_step_reward(
    prev_state: ExamState,
    next_state: ExamState,
    problems: list[Problem],
    action_name: str,
    reward_cfg: dict,
) -> float:
    prev_u = expected_utility(prev_state, problems)
    next_u = expected_utility(next_state, problems)
    base_gain = next_u - prev_u
    reward = base_gain
    if action_name == "solve_more":
        reward += _rw(reward_cfg, "solve_more", "penalty", default=0.0)
        if base_gain < _rw(reward_cfg, "solve_more", "low_marginal_gain_threshold", default=0.0):
            reward += _rw(reward_cfg, "solve_more", "low_marginal_gain_penalty", default=0.0)
        streak_threshold = _iw(reward_cfg, "solve_more", "streak", "threshold", default=0)
        if next_state.same_problem_streak > streak_threshold:
            # Flat constant penalty per step once over the threshold.
            # Do NOT multiply by extra_steps — that would make the penalty grow
            # quadratically with streak length and dominate all other reward signals.
            reward += _rw(reward_cfg, "solve_more", "streak", "penalty", default=0.0)

    if action_name == "next":
        reward += _rw(reward_cfg, "next", "penalty", default=0.0)
        reward += _next_transition_shaping(prev_state, problems, reward_cfg)
        if next_state.current_problem_idx not in prev_state.visit_order:
            reward += _coverage_bonus(prev_state, problems, reward_cfg)

    return float(reward)


def _coverage_fraction(state: ExamState) -> float:
    visited = sum(1 for p in state.progress if p.status != ProblemStatus.NOT_VISITED)
    return visited / max(len(state.progress), 1)


def compute_terminal_reward(
    state: ExamState,
    problems: list[Problem],
    reward_cfg: dict,
    *,
    timed_out: bool = False,
    step_limited: bool = False,
) -> float:
    reward = 0.0
    # Apply timeout_penalty whenever the agent ran out of resources —
    # either time expired or the episode was force-terminated by the step limit.
    if timed_out or step_limited:
        reward += _rw(reward_cfg, "terminal", "timeout_penalty", default=0.0)
    # Completion bonus scales proportionally with coverage fraction so the agent
    # always receives a gradient signal — visiting k/N problems yields
    # (k/N) * completion_bonus rather than 0 until all N are visited.
    coverage = _coverage_fraction(state)
    reward += coverage * _rw(reward_cfg, "terminal", "completion_bonus", default=0.0)
    return float(reward)
