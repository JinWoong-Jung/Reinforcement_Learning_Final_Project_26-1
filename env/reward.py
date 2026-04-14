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
    return float(sum(float(problem.score) * progress.effective_confidence(problem) for problem, progress in zip(problems, state.progress)))


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
    # Exit shaping should follow the confidence representation the agent can actually observe.
    # - subjective: answer_confidence
    # - objective: highest choice confidence
    confidence = float(prev_state.progress[current_idx].observable_confidence(problem))
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
        if (
            any(progress.status == ProblemStatus.NOT_VISITED for idx, progress in enumerate(prev_state.progress) if idx != current_idx)
            and confidence < _rw(exit_cfg, "hard", "defer_threshold", default=0.45)
        ):
            return _rw(exit_cfg, "hard", "defer_bonus", default=0.04)
        if confidence >= _rw(exit_cfg, "hard", "ready_threshold", default=0.35):
            return _rw(exit_cfg, "hard", "ready_bonus", default=0.06)
        return 0.0

    return 0.0


def _next_unvisited_from(prev_state: ExamState, current_idx: int) -> int | None:
    indices = list(range(current_idx + 1, len(prev_state.progress))) + list(range(0, current_idx))
    for idx in indices:
        if prev_state.progress[idx].status == ProblemStatus.NOT_VISITED:
            return idx
    return None


def _first_pass_next_shaping(prev_state: ExamState, next_state: ExamState, reward_cfg: dict) -> float:
    first_pass_cfg = _cfg(reward_cfg, "next", "first_pass", default={})
    current_idx = prev_state.current_problem_idx
    sequential_target = _next_unvisited_from(prev_state, current_idx)
    if sequential_target is None:
        return 0.0

    target_idx = next_state.current_problem_idx
    if target_idx in prev_state.visit_order:
        return _rw(first_pass_cfg, "revisit_penalty", default=0.0)

    if target_idx == sequential_target:
        return _rw(first_pass_cfg, "sequential_bonus", default=0.0)

    return 0.0


def _solve_more_confidence_gain(prev_state: ExamState, next_state: ExamState, problems: list[Problem]) -> float:
    problem_idx = prev_state.current_problem_idx
    problem = problems[problem_idx]
    prev_progress = prev_state.progress[problem_idx]
    next_progress = next_state.progress[problem_idx]
    if problem.problem_type == "objective":
        prev_conf = prev_progress.effective_confidence(problem)
        next_conf = next_progress.effective_confidence(problem)
    else:
        prev_conf = float(prev_progress.answer_confidence)
        next_conf = float(next_progress.answer_confidence)
    return float(next_conf - prev_conf)


def _low_marginal_gain_threshold(problem: Problem, reward_cfg: dict) -> float:
    nested_cfg = _cfg(reward_cfg, "solve_more", "low_marginal_gain", default={})
    if problem.problem_type == "objective":
        return _rw(
            {"value": nested_cfg.get("objective_threshold")} if isinstance(nested_cfg, dict) and "objective_threshold" in nested_cfg else None,
            "value",
            default=_rw(reward_cfg, "solve_more", "low_marginal_gain_threshold", default=0.0),
        )
    return _rw(
        {"value": nested_cfg.get("subjective_threshold")} if isinstance(nested_cfg, dict) and "subjective_threshold" in nested_cfg else None,
        "value",
        default=_rw(reward_cfg, "solve_more", "low_marginal_gain_threshold", default=0.0),
    )


def _low_marginal_gain_penalty(reward_cfg: dict) -> float:
    nested_cfg = _cfg(reward_cfg, "solve_more", "low_marginal_gain", default={})
    if isinstance(nested_cfg, dict) and "penalty" in nested_cfg:
        return float(nested_cfg["penalty"])
    return _rw(reward_cfg, "solve_more", "low_marginal_gain_penalty", default=0.0)


def _saturation_threshold(problem: Problem, reward_cfg: dict) -> float:
    saturation_cfg = _cfg(reward_cfg, "solve_more", "saturation", default={})
    if problem.problem_type == "objective":
        return float(_cfg(saturation_cfg, "objective", "threshold", default=1.1))
    return float(_cfg(saturation_cfg, "subjective", "threshold", default=1.1))


def _saturation_penalty(problem: Problem, reward_cfg: dict) -> float:
    saturation_cfg = _cfg(reward_cfg, "solve_more", "saturation", default={})
    if problem.problem_type == "objective":
        return float(_cfg(saturation_cfg, "objective", "penalty", default=0.0))
    return float(_cfg(saturation_cfg, "subjective", "penalty", default=0.0))


def _post_solve_confidence(next_state: ExamState, problems: list[Problem]) -> float:
    problem_idx = next_state.current_problem_idx
    problem = problems[problem_idx]
    progress = next_state.progress[problem_idx]
    if problem.problem_type == "objective":
        return float(progress.effective_confidence(problem))
    return float(progress.answer_confidence)


def _streak_penalty(next_state: ExamState, reward_cfg: dict) -> float:
    streak_cfg = _cfg(reward_cfg, "solve_more", "streak", default={})
    threshold = int(_cfg(streak_cfg, "threshold", default=0))
    if next_state.same_problem_streak <= threshold:
        return 0.0
    extra_steps = next_state.same_problem_streak - threshold
    scale = float(_cfg(streak_cfg, "extra_penalty_scale", default=0.0))
    cap = int(_cfg(streak_cfg, "max_extra_steps", default=extra_steps))
    effective_extra = min(max(extra_steps, 0), max(cap, 0))
    base_penalty = float(_cfg(streak_cfg, "penalty", default=0.0))
    return base_penalty + (scale * effective_extra)


def _topk_time_share(state: ExamState, k: int) -> float:
    problem_times = [float(p.time_spent_sec) for p in state.progress]
    total_time = float(sum(problem_times))
    if total_time <= 0:
        return 0.0
    topk = sum(sorted(problem_times, reverse=True)[:k])
    return float(topk / total_time)


def _concentration_penalty(state: ExamState, reward_cfg: dict) -> float:
    concentration_cfg = _cfg(reward_cfg, "terminal", "concentration", default={})
    penalty = 0.0
    top1_share = _topk_time_share(state, 1)
    top2_share = _topk_time_share(state, 2)

    top1_threshold = float(_cfg(concentration_cfg, "top1", "threshold", default=1.1))
    top1_scale = float(_cfg(concentration_cfg, "top1", "penalty_scale", default=0.0))
    if top1_share > top1_threshold:
        penalty += top1_scale * (top1_share - top1_threshold)

    top2_threshold = float(_cfg(concentration_cfg, "top2", "threshold", default=1.1))
    top2_scale = float(_cfg(concentration_cfg, "top2", "penalty_scale", default=0.0))
    if top2_share > top2_threshold:
        penalty += top2_scale * (top2_share - top2_threshold)

    return float(penalty)


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
        problem = problems[prev_state.current_problem_idx]
        confidence_gain = _solve_more_confidence_gain(prev_state, next_state, problems)
        reward += _rw(reward_cfg, "solve_more", "penalty", default=0.0)
        if confidence_gain < _low_marginal_gain_threshold(problem, reward_cfg):
            reward += _low_marginal_gain_penalty(reward_cfg)
        post_confidence = _post_solve_confidence(next_state, problems)
        if post_confidence >= _saturation_threshold(problem, reward_cfg):
            reward += _saturation_penalty(problem, reward_cfg)
        reward += _streak_penalty(next_state, reward_cfg)

    if action_name == "next":
        reward += _rw(reward_cfg, "next", "penalty", default=0.0)
        reward += _next_transition_shaping(prev_state, problems, reward_cfg)
        reward += _first_pass_next_shaping(prev_state, next_state, reward_cfg)
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
    reward += _concentration_penalty(state, reward_cfg)
    return float(reward)
