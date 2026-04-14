from __future__ import annotations

import copy
import unittest

from env.problem import Problem
from env.reward import compute_step_reward, compute_terminal_reward, expected_utility
from env.state import ExamState, ProblemProgress, ProblemStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PURE_REWARD_CFG: dict = {
    "next": {
        "penalty": 0.0,
        "new_problem_bonus": 0.0,
        "coverage_bonus_scale": 0.0,
        "coverage_bonus_power": 1.0,
        "first_pass": {"sequential_bonus": 0.0, "revisit_penalty": 0.0},
        "difficulty_exit": {
            "easy": {"low_conf_threshold": 0.0, "low_conf_penalty": 0.0, "ready_threshold": 1.1, "ready_bonus": 0.0},
            "mid":  {"low_conf_threshold": 0.0, "low_conf_penalty": 0.0, "ready_threshold": 1.1, "ready_bonus": 0.0},
            "hard": {"defer_threshold": 0.0,    "defer_bonus": 0.0,      "ready_threshold": 1.1, "ready_bonus": 0.0},
        },
    },
    "solve_more": {
        "penalty": 0.0,
        "low_marginal_gain": {
            "subjective_threshold": 0.0,
            "objective_threshold": 0.0,
            "penalty": 0.0,
        },
        "saturation": {
            "subjective": {"threshold": 1.1, "penalty": 0.0},
            "objective": {"threshold": 1.1, "penalty": 0.0},
        },
        "streak": {"threshold": 9999, "penalty": 0.0, "extra_penalty_scale": 0.0, "max_extra_steps": 0},
    },
    "terminal": {
        "timeout_penalty": 0.0,
        "completion_bonus": 0.0,
        "concentration": {
            "top1": {"threshold": 1.1, "penalty_scale": 0.0},
            "top2": {"threshold": 1.1, "penalty_scale": 0.0},
        },
    },
}


def _obj_problem(pid: int = 0, score: int = 3) -> Problem:
    return Problem(
        pid=pid,
        difficulty_level="중",
        difficulty=0.5,
        score=score,
        error_rate=0.4,
        problem_type="objective",
        actual_answer=1,
        choice_rate={"1": 0.4, "2": 0.3, "3": 0.1, "4": 0.1, "5": 0.1},
        correct_rate=0.5,
    )


def _subj_problem(pid: int = 1, score: int = 4) -> Problem:
    return Problem(
        pid=pid,
        difficulty_level="중",
        difficulty=0.5,
        score=score,
        error_rate=0.4,
        problem_type="subjective",
        correct_rate=0.5,
    )


def _make_state(problems: list[Problem], current_idx: int, confidences: list[float]) -> ExamState:
    progress = []
    for i, (problem, conf) in enumerate(zip(problems, confidences)):
        p = ProblemProgress(status=ProblemStatus.IN_PROGRESS if i == current_idx else ProblemStatus.NOT_VISITED)
        p.sync_from_scalar(problem, conf)
        progress.append(p)
    state = ExamState(
        remaining_time_sec=300.0,
        current_problem_idx=current_idx,
        progress=progress,
        total_score=0.0,
        visit_order=[current_idx],
    )
    state.total_score = expected_utility(state, problems)
    return state


# ---------------------------------------------------------------------------
# Pure reward tests
# ---------------------------------------------------------------------------

class PureRewardTests(unittest.TestCase):
    """With all shaping off, step reward == delta expected_utility."""

    def _assert_pure(self, prev_state, next_state, problems, action_name):
        expected_delta = expected_utility(next_state, problems) - expected_utility(prev_state, problems)
        reward = compute_step_reward(
            prev_state=prev_state,
            next_state=next_state,
            problems=problems,
            action_name=action_name,
            reward_cfg=_PURE_REWARD_CFG,
        )
        self.assertAlmostEqual(reward, expected_delta, places=9)

    def test_solve_more_objective_pure_reward(self):
        problem = _obj_problem()
        prev = _make_state([problem], 0, [0.30])
        next_ = _make_state([problem], 0, [0.55])
        self._assert_pure(prev, next_, [problem], "solve_more")

    def test_solve_more_subjective_pure_reward(self):
        problem = _subj_problem()
        prev = _make_state([problem], 0, [0.20])
        next_ = _make_state([problem], 0, [0.40])
        self._assert_pure(prev, next_, [problem], "solve_more")

    def test_next_pure_reward(self):
        p0 = _obj_problem(pid=0)
        p1 = _subj_problem(pid=1)
        prev = _make_state([p0, p1], 0, [0.60, 0.30])
        next_ = copy.deepcopy(prev)
        next_.progress[0].status = ProblemStatus.MOVED_ON
        next_.progress[1].status = ProblemStatus.IN_PROGRESS
        next_.current_problem_idx = 1
        self._assert_pure(prev, next_, [p0, p1], "next")

    def test_zero_confidence_gain_gives_zero_reward(self):
        problem = _obj_problem()
        prev = _make_state([problem], 0, [0.70])
        next_ = copy.deepcopy(prev)  # no change in confidence
        reward = compute_step_reward(prev, next_, [problem], "solve_more", _PURE_REWARD_CFG)
        self.assertAlmostEqual(reward, 0.0, places=9)

    def test_pure_reward_is_additive_across_problems(self):
        p0 = _obj_problem(pid=0, score=2)
        p1 = _subj_problem(pid=1, score=3)
        prev = _make_state([p0, p1], 0, [0.40, 0.20])
        next_ = _make_state([p0, p1], 0, [0.60, 0.20])  # only p0 changes
        reward = compute_step_reward(prev, next_, [p0, p1], "solve_more", _PURE_REWARD_CFG)
        expected = 2 * (0.60 - 0.40)  # score_0 * delta_conf_0; p1 unchanged
        self.assertAlmostEqual(reward, expected, places=6)


class TerminalRewardTests(unittest.TestCase):
    """Terminal reward is zero when all shaping is off."""

    def _make_terminal_state(self) -> tuple[ExamState, list[Problem]]:
        p0 = _obj_problem()
        p1 = _subj_problem()
        state = _make_state([p0, p1], 0, [0.6, 0.4])
        state.remaining_time_sec = 0.0
        return state, [p0, p1]

    def test_terminal_reward_zero_with_all_shaping_off(self):
        state, problems = self._make_terminal_state()
        reward = compute_terminal_reward(state, problems, _PURE_REWARD_CFG, timed_out=True)
        self.assertAlmostEqual(reward, 0.0, places=9)

    def test_timeout_penalty_is_applied_when_nonzero(self):
        state, problems = self._make_terminal_state()
        cfg = {**_PURE_REWARD_CFG, "terminal": {**_PURE_REWARD_CFG["terminal"], "timeout_penalty": -2.0}}
        reward = compute_terminal_reward(state, problems, cfg, timed_out=True)
        self.assertAlmostEqual(reward, -2.0, places=6)

    def test_completion_bonus_scales_with_coverage(self):
        p0 = _obj_problem(pid=0)
        p1 = _subj_problem(pid=1)
        # Visit only first problem
        state = _make_state([p0, p1], 0, [0.6, 0.0])
        state.progress[1].status = ProblemStatus.NOT_VISITED
        cfg = {**_PURE_REWARD_CFG, "terminal": {**_PURE_REWARD_CFG["terminal"], "completion_bonus": 10.0}}
        reward = compute_terminal_reward(state, [p0, p1], cfg, timed_out=True)
        # coverage = 1/2, so bonus = 0.5 * 10 = 5.0
        self.assertAlmostEqual(reward, 5.0, places=6)


class ShapingTermsTests(unittest.TestCase):
    """Shaping terms must only fire when explicitly configured."""

    def test_solve_more_penalty_is_additive(self):
        problem = _subj_problem()
        prev = _make_state([problem], 0, [0.20])
        next_ = _make_state([problem], 0, [0.35])
        cfg = {**_PURE_REWARD_CFG, "solve_more": {**_PURE_REWARD_CFG["solve_more"], "penalty": -0.01}}
        delta = expected_utility(next_, [problem]) - expected_utility(prev, [problem])
        reward = compute_step_reward(prev, next_, [problem], "solve_more", cfg)
        self.assertAlmostEqual(reward, delta - 0.01, places=6)

    def test_next_penalty_is_additive(self):
        p0 = _obj_problem(pid=0)
        p1 = _subj_problem(pid=1)
        prev = _make_state([p0, p1], 0, [0.60, 0.30])
        next_ = copy.deepcopy(prev)
        next_.progress[0].status = ProblemStatus.MOVED_ON
        next_.progress[1].status = ProblemStatus.IN_PROGRESS
        next_.current_problem_idx = 1
        cfg = {**_PURE_REWARD_CFG, "next": {**_PURE_REWARD_CFG["next"], "penalty": -0.05}}
        delta = expected_utility(next_, [p0, p1]) - expected_utility(prev, [p0, p1])
        reward = compute_step_reward(prev, next_, [p0, p1], "next", cfg)
        self.assertAlmostEqual(reward, delta - 0.05, places=6)
