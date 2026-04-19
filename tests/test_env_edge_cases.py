"""Edge case tests for ExamStrategyEnv.

Covers:
  - remaining_time=0 → episode terminates immediately
  - step_count >= max_steps → episode truncates
  - single-problem exam runs to completion
  - all-objective exam runs without error
  - all-subjective exam runs without error
  - all-same-difficulty exam runs without error
  - solve_more when remaining_time=0 does nothing to state
  - high-score hard problem: confidence rises (slowly but stays in [floor, 1])
  - episode always terminates within max_steps
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from env.exam_env import ExamStrategyEnv
from env.dynamics import confidence_curve, guessing_prob
from env.problem import Problem
from env.state import ProblemStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob(
    pid: int,
    difficulty_level: str,
    difficulty: float,
    score: int,
    correct_rate: float,
    problem_type: str = "objective",
    actual_answer: int = 1,
    num_choices: int = 5,
) -> dict:
    if problem_type == "objective":
        other = (1.0 - correct_rate) / max(num_choices - 1, 1)
        choice_rate = {
            str(i + 1): (correct_rate if i + 1 == actual_answer else other)
            for i in range(num_choices)
        }
    else:
        choice_rate = {"correct": correct_rate}
    return {
        "pid": pid,
        "difficulty_level": difficulty_level,
        "difficulty": difficulty,
        "score": score,
        "correct_rate": correct_rate,
        "error_rate": round(1.0 - correct_rate, 6),
        "problem_type": problem_type,
        "actual_answer": actual_answer,
        "choice_rate": choice_rate,
    }


def _write_exam_json(problems: list[dict], total_time_sec: float, tmpdir: str) -> str:
    data = {
        "exam_id": "test_exam",
        "subject": "test",
        "total_time_sec": total_time_sec,
        "problems": problems,
    }
    path = os.path.join(tmpdir, "exam.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _make_env(
    problems: list[dict],
    total_time_sec: float,
    switch_time_sec: float = 10.0,
    action_time_unit_sec: float = 30.0,
    max_steps: int | None = None,
    tmpdir: str | None = None,
    seed: int = 0,
    owned_tmpdir: list[tempfile.TemporaryDirectory] | None = None,
) -> ExamStrategyEnv:
    td = tempfile.TemporaryDirectory()
    if owned_tmpdir is not None:
        owned_tmpdir.append(td)
    path = _write_exam_json(problems, total_time_sec, td.name)
    exam_cfg: dict = {
        "action_time_unit_sec": action_time_unit_sec,
        "switch_time_sec": switch_time_sec,
        "randomize_start_problem": False,
    }
    if max_steps is not None:
        exam_cfg["max_steps"] = max_steps
    cfg = {
        "exam": exam_cfg,
        "data": {"exam_path": path},
        "reward": {},
        "dynamics": {},
    }
    return ExamStrategyEnv(config=cfg, random_seed=seed, fixed_student_level="mid")


def _run_until_done(env: ExamStrategyEnv, solve_only: bool = False, max_iters: int = 100_000) -> int:
    """Run an episode using solve_more-only or mixed actions. Returns step count."""
    env.reset(seed=0)
    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_iters:
        if solve_only or env.state.current_problem_idx == -1:
            action = env.encode_solve_more_action()
        else:
            action = env.encode_solve_more_action()
        _, _, done, truncated, _ = env.step(action)
        steps += 1
    return steps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTerminationConditions(unittest.TestCase):
    """Episode must terminate via time-out or step-count limit."""

    def _simple_problems(self) -> list[dict]:
        return [
            _prob(1, "하", 0.10, 3, 0.90),
            _prob(2, "중", 0.50, 3, 0.50),
            _prob(3, "상", 0.80, 4, 0.20),
        ]

    def test_time_exhaustion_terminates(self):
        """When remaining_time reaches 0, terminated=True is returned."""
        env = _make_env(self._simple_problems(), total_time_sec=60.0, action_time_unit_sec=30.0)
        env.reset(seed=0)
        done = truncated = False
        for _ in range(500):
            obs, reward, done, truncated, info = env.step(env.encode_solve_more_action())
            if done or truncated:
                break
        self.assertTrue(done or truncated, "Episode should have terminated within 500 steps")
        self.assertLessEqual(env.state.remaining_time_sec, 0.0)

    def test_step_count_limit_truncates(self):
        """When max_steps is exceeded, the episode terminates (done=True via step_count)."""
        env = _make_env(self._simple_problems(), total_time_sec=10000.0, max_steps=5)
        env.reset(seed=0)
        done = truncated = False
        for _ in range(20):
            _, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            if done or truncated:
                break
        self.assertTrue(done or truncated, "Episode must terminate once step count reaches max_steps")
        self.assertGreaterEqual(env.state.step_count, 5)

    def test_already_done_env_returns_done(self):
        """Stepping on an already-terminated env returns (obs, 0, True, False, ...)."""
        env = _make_env(self._simple_problems(), total_time_sec=30.0, action_time_unit_sec=30.0)
        env.reset(seed=0)
        # Drain all time
        for _ in range(20):
            _, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            if done or truncated:
                break
        # One more step on a finished env
        _, reward, done2, _, info = env.step(env.encode_solve_more_action())
        self.assertTrue(done2)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info.get("reason"), "already_done")

    def test_episode_always_terminates_within_max_steps(self):
        """An episode must always terminate within max_steps regardless of actions."""
        max_steps = 50
        env = _make_env(self._simple_problems(), total_time_sec=10000.0, max_steps=max_steps)
        env.reset(seed=42)
        done = truncated = False
        count = 0
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            count += 1
        self.assertLessEqual(count, max_steps + 1)


class TestSingleProblemEnv(unittest.TestCase):
    """A one-problem exam must run to completion without error."""

    def _single_problem_env(self) -> ExamStrategyEnv:
        probs = [_prob(1, "중", 0.50, 4, 0.50)]
        return _make_env(probs, total_time_sec=120.0, action_time_unit_sec=30.0)

    def test_single_problem_episode_completes(self):
        env = self._single_problem_env()
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)
        self.assertIsNotNone(env.state)

    def test_single_problem_next_action_does_not_crash(self):
        """Sending a 'next' action on a single-problem env should not raise."""
        env = self._single_problem_env()
        env.reset(seed=0)
        # action_type=1, target=0: only valid index; env redirects to itself
        action = np.array([1, 0], dtype=np.int64)
        obs, reward, done, truncated, info = env.step(action)
        self.assertIsNotNone(obs)

    def test_single_problem_obs_shape(self):
        env = self._single_problem_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)


class TestAllObjectiveProblems(unittest.TestCase):
    """An exam with only objective problems must run without error."""

    def _all_objective_env(self) -> ExamStrategyEnv:
        probs = [
            _prob(1, "하", 0.10, 2, 0.90),
            _prob(2, "중하", 0.25, 3, 0.75),
            _prob(3, "중", 0.50, 3, 0.50),
            _prob(4, "상", 0.75, 4, 0.25),
        ]
        return _make_env(probs, total_time_sec=600.0)

    def test_all_objective_episode_runs(self):
        env = self._all_objective_env()
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)

    def test_all_objective_confidence_in_range(self):
        env = self._all_objective_env()
        env.reset(seed=0)
        for _ in range(30):
            env.step(env.encode_solve_more_action())
        for i, progress in enumerate(env.state.progress):
            for c in progress.choice_confidences:
                self.assertGreaterEqual(c, 0.0, f"problem {i} choice confidence < 0")
                self.assertLessEqual(c, 1.0, f"problem {i} choice confidence > 1")

    def test_all_objective_choice_confidences_sum_to_one(self):
        """After solving, choice_confidences must sum to ~1.0."""
        env = self._all_objective_env()
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.encode_solve_more_action())
        for i, (progress, problem) in enumerate(zip(env.state.progress, env.problems)):
            if progress.choice_confidences:
                total = sum(progress.choice_confidences)
                self.assertAlmostEqual(total, 1.0, places=5, msg=f"problem {i}: choice confidences sum={total}")


class TestAllSubjectiveProblems(unittest.TestCase):
    """An exam with only subjective problems must run without error."""

    def _all_subjective_env(self) -> ExamStrategyEnv:
        probs = [
            _prob(1, "하", 0.10, 3, 0.88, problem_type="subjective"),
            _prob(2, "중", 0.50, 4, 0.50, problem_type="subjective"),
            _prob(3, "최상", 0.90, 4, 0.10, problem_type="subjective"),
        ]
        return _make_env(probs, total_time_sec=600.0)

    def test_all_subjective_episode_runs(self):
        env = self._all_subjective_env()
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)

    def test_all_subjective_no_choice_confidences(self):
        """Subjective problems have no choice_confidences list."""
        env = self._all_subjective_env()
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.encode_solve_more_action())
        for i, progress in enumerate(env.state.progress):
            self.assertEqual(
                progress.choice_confidences, [],
                msg=f"problem {i} (subjective) should have empty choice_confidences",
            )

    def test_all_subjective_floor_at_t0(self):
        """At t=0, subjective answer_confidence must be >= subjective floor (0.0)."""
        env = self._all_subjective_env()
        env.reset(seed=0)
        for progress in env.state.progress:
            self.assertGreaterEqual(progress.answer_confidence, 0.0 - 1e-9)


class TestAllIdenticalProblems(unittest.TestCase):
    """An exam where every problem is identical should run without error."""

    def _identical_env(self) -> ExamStrategyEnv:
        probs = [_prob(i + 1, "중", 0.50, 3, 0.50) for i in range(5)]
        return _make_env(probs, total_time_sec=600.0)

    def test_identical_problems_episode_runs(self):
        env = self._identical_env()
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)

    def test_identical_problems_obs_normalized(self):
        """All observation values must be in [0, 1]."""
        env = self._identical_env()
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            obs, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            if done or truncated:
                break
        self.assertTrue(
            np.all(obs >= -1e-6) and np.all(obs <= 1.0 + 1e-6),
            msg=f"Observation out of [0,1]: min={obs.min():.4f} max={obs.max():.4f}",
        )


class TestSolveMoreWhenTimeIsZero(unittest.TestCase):
    """solve_more with no remaining time must not change the state's time_spent."""

    def test_solve_more_at_zero_time_does_not_increase_time_spent(self):
        probs = [_prob(1, "중", 0.50, 3, 0.50), _prob(2, "중", 0.50, 3, 0.50)]
        env = _make_env(probs, total_time_sec=60.0, action_time_unit_sec=30.0)
        env.reset(seed=0)

        # Drain all time
        for _ in range(10):
            _, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            if done or truncated:
                break

        time_spent_before = [p.time_spent_sec for p in env.state.progress]
        env.step(env.encode_solve_more_action())  # step after done
        time_spent_after = [p.time_spent_sec for p in env.state.progress]

        self.assertEqual(
            time_spent_before, time_spent_after,
            "time_spent should not change after time is exhausted",
        )


class TestHighScoreHardProblem(unittest.TestCase):
    """A very hard high-score problem: confidence must stay in [floor, 1] and rise slowly."""

    def _hard_problem(self) -> Problem:
        return Problem(
            pid=99,
            difficulty_level="최상",
            difficulty=0.95,
            score=4,
            error_rate=0.94,
            problem_type="subjective",
            correct_rate=0.06,
        )

    def test_confidence_in_valid_range_for_hard_problem(self):
        from env.student import create_level_profile
        student = create_level_profile("mid")
        problem = self._hard_problem()
        floor = guessing_prob(problem)
        for t in [0, 30, 60, 120, 300, 600, 1200]:
            conf = confidence_curve(problem, student, float(t))
            self.assertGreaterEqual(conf, floor - 1e-9, msg=f"t={t}: conf={conf:.4f} < floor={floor:.4f}")
            self.assertLessEqual(conf, 1.0 + 1e-9, msg=f"t={t}: conf={conf:.4f} > 1")

    def test_hard_problem_confidence_below_easy_at_same_time(self):
        """At the same elapsed time, hard problem confidence < easy problem confidence."""
        from env.student import create_level_profile
        student = create_level_profile("mid")
        hard = self._hard_problem()
        easy = Problem(
            pid=1, difficulty_level="하", difficulty=0.05, score=2,
            error_rate=0.05, problem_type="objective", correct_rate=0.95,
            choice_rate={"1": 0.95, "2": 0.01, "3": 0.01, "4": 0.02, "5": 0.01},
            actual_answer=1,
        )
        for t in [30.0, 60.0, 120.0]:
            conf_hard = confidence_curve(hard, student, t)
            conf_easy = confidence_curve(easy, student, t)
            self.assertLess(
                conf_hard, conf_easy,
                msg=f"t={t}s: hard ({conf_hard:.4f}) should be below easy ({conf_easy:.4f})",
            )

    def test_hard_problem_in_env_does_not_crash(self):
        probs = [
            _prob(1, "최상", 0.95, 4, 0.06, problem_type="subjective"),
        ]
        env = _make_env(probs, total_time_sec=300.0)
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)


class TestObservationSpace(unittest.TestCase):
    """Observation must always match the declared observation_space shape and range."""

    def _make_mixed_env(self) -> ExamStrategyEnv:
        probs = [
            _prob(1, "하", 0.10, 2, 0.90),
            _prob(2, "중", 0.50, 3, 0.50),
            _prob(3, "상", 0.80, 4, 0.20, problem_type="subjective"),
        ]
        return _make_env(probs, total_time_sec=300.0)

    def test_reset_obs_shape_matches_space(self):
        env = self._make_mixed_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_step_obs_shape_consistent(self):
        env = self._make_mixed_env()
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            obs, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            self.assertEqual(obs.shape, env.observation_space.shape)
            if done or truncated:
                break

    def test_obs_dtype_is_float32(self):
        env = self._make_mixed_env()
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_values_in_unit_interval(self):
        env = self._make_mixed_env()
        obs, _ = env.reset(seed=0)
        for _ in range(20):
            obs, _, done, truncated, _ = env.step(env.encode_solve_more_action())
            if done or truncated:
                break
        self.assertTrue(np.all(obs >= -1e-6))
        self.assertTrue(np.all(obs <= 1.0 + 1e-6))


class TestSwitchCostEdgeCases(unittest.TestCase):
    """Switch cost must be deducted correctly even at extreme values."""

    def test_zero_switch_cost_does_not_crash(self):
        probs = [_prob(1, "중", 0.5, 3, 0.5), _prob(2, "중", 0.5, 3, 0.5)]
        env = _make_env(probs, total_time_sec=300.0, switch_time_sec=0.0)
        env.reset(seed=0)
        action = np.array([1, 1], dtype=np.int64)
        obs, _, _, _, _ = env.step(action)
        self.assertIsNotNone(obs)

    def test_large_switch_cost_drains_time(self):
        """With switch_cost=total_time, a single 'next' action exhausts remaining time."""
        probs = [_prob(1, "중", 0.5, 3, 0.5), _prob(2, "중", 0.5, 3, 0.5)]
        env = _make_env(probs, total_time_sec=100.0, switch_time_sec=100.0)
        env.reset(seed=0)
        action = np.array([1, 1], dtype=np.int64)
        _, _, done, truncated, _ = env.step(action)
        self.assertLessEqual(env.state.remaining_time_sec, 0.0)


class TestToyExamFiles(unittest.TestCase):
    """Smoke tests that the checked-in toy JSON files load and produce valid episodes."""

    def _toy_cfg(self, path: str) -> dict:
        return {
            "exam": {
                "action_time_unit_sec": 30.0,
                "switch_time_sec": 10.0,
                "randomize_start_problem": False,
            },
            "data": {"exam_path": path},
            "reward": {},
            "dynamics": {},
        }

    def _toy_path(self, filename: str) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(project_root, "data", filename)

    def test_toy_3prob_loads_and_runs(self):
        path = self._toy_path("toy_3prob.json")
        if not os.path.exists(path):
            self.skipTest("toy_3prob.json not found")
        env = ExamStrategyEnv(config=self._toy_cfg(path), random_seed=0, fixed_student_level="mid")
        obs, info = env.reset(seed=0)
        self.assertEqual(env.num_problems, 3)
        self.assertEqual(obs.shape, env.observation_space.shape)
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)

    def test_toy_5prob_loads_and_runs(self):
        path = self._toy_path("toy_5prob.json")
        if not os.path.exists(path):
            self.skipTest("toy_5prob.json not found")
        env = ExamStrategyEnv(config=self._toy_cfg(path), random_seed=0, fixed_student_level="mid")
        obs, info = env.reset(seed=0)
        self.assertEqual(env.num_problems, 5)
        self.assertEqual(obs.shape, env.observation_space.shape)
        steps = _run_until_done(env)
        self.assertGreater(steps, 0)

    def test_toy_3prob_greedy_beats_random(self):
        """Over many episodes, greedy should score higher than random on toy_3prob."""
        from agents.heuristic_agents import run_heuristic_episode
        path = self._toy_path("toy_3prob.json")
        if not os.path.exists(path):
            self.skipTest("toy_3prob.json not found")
        cfg = self._toy_cfg(path)
        greedy_scores, random_scores = [], []
        for seed in range(30):
            env = ExamStrategyEnv(config=cfg, random_seed=seed, fixed_student_level="mid")
            greedy_scores.append(run_heuristic_episode(env, "marginal_gain_greedy", reset_seed=seed).total_score)
            env2 = ExamStrategyEnv(config=cfg, random_seed=seed, fixed_student_level="mid")
            random_scores.append(run_heuristic_episode(env2, "random", reset_seed=seed).total_score)
        mean_greedy = sum(greedy_scores) / len(greedy_scores)
        mean_random = sum(random_scores) / len(random_scores)
        self.assertGreater(
            mean_greedy, mean_random,
            msg=f"Greedy ({mean_greedy:.3f}) should beat random ({mean_random:.3f}) on toy_3prob",
        )


if __name__ == "__main__":
    unittest.main()
