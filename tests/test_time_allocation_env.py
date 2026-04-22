"""Tests for TimeAllocationEnv.

Covers:
  - obs shape and action space at reset
  - deterministic replay: same seed → identical trajectory
  - step reward equals delta expected_score
  - total allocated time never exceeds budget
  - coverage_fraction = fraction of problems that received any time
  - reserve_switch_time deducts overhead correctly
  - shuffle_problem_order_on_reset: pid list changes but env is valid
  - episode terminates when budget is exhausted
  - greedy_marginal_gain baseline produces valid episode stats
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from env.time_allocation_env import TimeAllocationEnv
from env.dynamics import expected_total_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prob(
    pid: int,
    difficulty_level: str = "중",
    difficulty: float = 0.5,
    score: int = 3,
    correct_rate: float = 0.5,
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


def _make_exam_json(problems: list[dict], total_time_sec: float, tmpdir: str) -> str:
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
    total_time_sec: float = 300.0,
    action_time_unit_sec: float = 30.0,
    switch_time_sec: float = 10.0,
    reserve_switch_time: bool = False,
    shuffle: bool = False,
    max_steps: int | None = None,
    seed: int = 0,
    student_fixed_id: str | None = None,
    score_bonus_scale: float = 1.0,
    allocation_prior_cfg: dict | None = None,
    owned_dirs: list | None = None,
) -> TimeAllocationEnv:
    td = tempfile.TemporaryDirectory()
    if owned_dirs is not None:
        owned_dirs.append(td)
    path = _make_exam_json(problems, total_time_sec, td.name)
    exam_cfg: dict = {
        "action_time_unit_sec": action_time_unit_sec,
        "switch_time_sec": switch_time_sec,
        "reserve_switch_time": reserve_switch_time,
        "shuffle_problem_order_on_reset": shuffle,
    }
    if max_steps is not None:
        exam_cfg["max_steps"] = max_steps
    # Use fixed_level to avoid needing student JSON files in tests
    student_cfg: dict = {"fixed_level": student_fixed_id if student_fixed_id else "mid"}
    reward_cfg = {
        "terminal": {
            "score_bonus_scale": score_bonus_scale,
            "completion_bonus": 0.0,
            "timeout_penalty": 0.0,
            "concentration": {
                "top1": {"threshold": 1.1, "penalty_scale": 0.0},
                "top2": {"threshold": 1.1, "penalty_scale": 0.0},
            },
        }
    }
    if allocation_prior_cfg is not None:
        reward_cfg["allocation"] = {"difficulty_time_prior": allocation_prior_cfg}

    cfg = {
        "exam": exam_cfg,
        "reward": reward_cfg,
        "student": student_cfg,
        "dynamics": {},
        "data": {
            "exam_path": path,
            "student_path": os.path.join(
                os.path.dirname(__file__), "..", "data", "theta_students.json"
            ),
        },
        "evaluation": {
            "solved": {
                "subjective_conf_threshold": 0.5,
                "objective_conf_threshold": 0.5,
                "objective_margin_threshold": 0.05,
            }
        },
    }
    env = TimeAllocationEnv(config=cfg, random_seed=seed)
    # Keep the temp directory alive by storing it on the env
    env._test_td = td  # type: ignore[attr-defined]
    return env


# 5 problems: mix of objective and subjective
_PROBLEMS_5 = [
    _prob(1, "하",  0.1, 2, 0.9, "objective"),
    _prob(2, "중",  0.5, 3, 0.5, "objective"),
    _prob(3, "상",  0.8, 4, 0.2, "objective"),
    _prob(4, "최상", 1.0, 5, 0.1, "subjective"),
    _prob(5, "중하", 0.3, 2, 0.7, "subjective"),
]


class TestObsAndActionSpace(unittest.TestCase):
    def test_obs_shape(self):
        env = _make_env(_PROBLEMS_5)
        obs, _ = env.reset(seed=0)
        expected_dim = 1 + len(_PROBLEMS_5) * 11
        self.assertEqual(obs.shape, (expected_dim,))
        self.assertEqual(obs.dtype, np.float32)

    def test_obs_space_matches(self):
        env = _make_env(_PROBLEMS_5)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_action_space_is_discrete(self):
        env = _make_env(_PROBLEMS_5)
        self.assertEqual(env.action_space.n, len(_PROBLEMS_5))

    def test_obs_values_in_range(self):
        env = _make_env(_PROBLEMS_5)
        obs, _ = env.reset(seed=0)
        self.assertTrue(np.all(obs >= -1e-6))
        self.assertTrue(np.all(obs <= 1.0 + 1e-6))


class TestDeterministicReplay(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        dirs: list = []
        env1 = _make_env(_PROBLEMS_5, seed=42, owned_dirs=dirs)
        env2 = _make_env(_PROBLEMS_5, seed=42, owned_dirs=dirs)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        actions = [2, 0, 3, 1, 4, 2, 0, 3, 1, 4]
        for a in actions:
            o1, r1, d1, t1, _ = env1.step(a)
            o2, r2, d2, t2, _ = env2.step(a)
            np.testing.assert_array_almost_equal(o1, o2)
            self.assertAlmostEqual(r1, r2, places=8)
            self.assertEqual(d1, d2)
            self.assertEqual(t1, t2)


class TestStepRewardEqualsDeltaScore(unittest.TestCase):
    def test_step_reward_equals_delta_expected_score(self):
        env = _make_env(_PROBLEMS_5, score_bonus_scale=0.0)
        env.reset(seed=0)
        for action in range(env.num_problems):
            assert env.state is not None
            score_before = float(env.state.total_score)
            _, reward, done, _, _ = env.step(action)
            if done:
                break
            score_after = float(env.state.total_score)
            self.assertAlmostEqual(reward, score_after - score_before, places=6)

    def test_difficulty_time_prior_penalizes_over_target_time(self):
        env = _make_env(
            [_prob(1, "하", 0.1, 2, 0.9, "objective")],
            total_time_sec=90.0,
            action_time_unit_sec=30.0,
            score_bonus_scale=0.0,
            allocation_prior_cfg={
                "enabled": True,
                "target_sec": {"하": 30},
                "under_target_bonus_scale": {"하": 0.0},
                "over_target_penalty_scale": {"하": -0.5},
                "over_target_penalty_power": 1.0,
            },
        )
        env.reset(seed=0)
        env.step(0)
        assert env.state is not None
        score_before = float(env.state.total_score)
        _, reward, done, _, _ = env.step(0)
        self.assertFalse(done)
        score_after = float(env.state.total_score)
        self.assertAlmostEqual(reward, (score_after - score_before) - 0.5, places=6)


class TestBudgetNeverExceeded(unittest.TestCase):
    def test_total_time_within_budget(self):
        env = _make_env(_PROBLEMS_5, total_time_sec=600.0)
        env.reset(seed=7)
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.n - 1  # always last problem
            _, _, done, truncated, _ = env.step(action)
        assert env.state is not None
        total_spent = sum(float(p.time_spent_sec) for p in env.state.progress)
        # total_spent + remaining ≈ initial budget
        initial_budget = env._compute_available_time(
            float(env.exam_cfg.get("total_time_sec", 600.0)), env.num_problems
        )
        self.assertLessEqual(total_spent, initial_budget + 1e-6)
        self.assertGreaterEqual(float(env.state.remaining_time_sec), -1e-6)

    def test_reserve_switch_time_reduces_budget(self):
        dirs: list = []
        env_no_reserve = _make_env(_PROBLEMS_5, total_time_sec=600.0,
                                   reserve_switch_time=False, owned_dirs=dirs)
        env_reserve = _make_env(_PROBLEMS_5, total_time_sec=600.0,
                                reserve_switch_time=True, switch_time_sec=10.0,
                                owned_dirs=dirs)
        obs_no, info_no = env_no_reserve.reset(seed=0)
        obs_res, info_res = env_reserve.reset(seed=0)
        self.assertAlmostEqual(info_no["available_time_sec"], 600.0, places=4)
        expected = 600.0 - (len(_PROBLEMS_5) - 1) * 10.0
        self.assertAlmostEqual(info_res["available_time_sec"], expected, places=4)


class TestCoverageFraction(unittest.TestCase):
    def test_unvisited_problems_reduce_coverage(self):
        # Only enough budget for 3 actions on 5 problems
        env = _make_env(_PROBLEMS_5, total_time_sec=90.0, action_time_unit_sec=30.0,
                        max_steps=3)
        env.reset(seed=0)
        done = truncated = False
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(0)  # only problem 0
        assert env.state is not None
        cov = env.state.coverage_fraction()
        # Only problem 0 was worked on → coverage = 1/5
        self.assertAlmostEqual(cov, 1.0 / len(_PROBLEMS_5), places=5)

    def test_full_coverage_when_all_touched(self):
        # Budget exactly = N * action_unit
        n = len(_PROBLEMS_5)
        env = _make_env(_PROBLEMS_5, total_time_sec=float(n * 30), action_time_unit_sec=30.0)
        env.reset(seed=0)
        done = truncated = False
        step = 0
        while not (done or truncated):
            _, _, done, truncated, _ = env.step(step % n)
            step += 1
        assert env.state is not None
        self.assertAlmostEqual(env.state.coverage_fraction(), 1.0, places=5)


class TestShuffleProblemOrder(unittest.TestCase):
    def test_pids_change_across_resets(self):
        env = _make_env(_PROBLEMS_5, shuffle=True)
        pids_seen: set[tuple] = set()
        for s in range(20):
            _, info = env.reset(seed=s)
            pids_seen.add(tuple(info["problem_pids"]))
        # With 5! = 120 permutations, we expect multiple distinct orders in 20 tries
        self.assertGreater(len(pids_seen), 1)

    def test_env_valid_after_shuffle(self):
        env = _make_env(_PROBLEMS_5, shuffle=True)
        env.reset(seed=99)
        done = truncated = False
        for _ in range(10):
            if done or truncated:
                break
            _, _, done, truncated, _ = env.step(0)


class TestEpisodeTermination(unittest.TestCase):
    def test_episode_ends_when_budget_exhausted(self):
        env = _make_env(_PROBLEMS_5, total_time_sec=90.0, action_time_unit_sec=30.0)
        env.reset(seed=0)
        rewards = []
        done = truncated = False
        while not (done or truncated):
            _, r, done, truncated, _ = env.step(0)
            rewards.append(r)
        self.assertTrue(done or truncated)
        assert env.state is not None
        self.assertLessEqual(env.state.remaining_time_sec, 1e-6)

    def test_already_done_returns_true(self):
        env = _make_env(_PROBLEMS_5, total_time_sec=30.0, action_time_unit_sec=30.0)
        env.reset(seed=0)
        env.step(0)  # exhausts budget
        _, _, done, _, _ = env.step(1)
        self.assertTrue(done)


class TestGreedyBaseline(unittest.TestCase):
    def test_greedy_marginal_gain_runs_to_completion(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from agents.heuristic_agents import (
            allocation_policy_greedy_marginal_gain,
            evaluate_allocation_policy,
        )

        def factory():
            return _make_env(_PROBLEMS_5, total_time_sec=300.0)

        result = evaluate_allocation_policy(factory, "greedy_marginal_gain", episodes=5, seed=0)
        self.assertEqual(result["policy"], "greedy_marginal_gain")
        self.assertGreater(result["mean_score"], 0.0)
        self.assertLessEqual(result["mean_coverage_fraction"], 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
