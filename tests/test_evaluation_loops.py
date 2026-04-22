from __future__ import annotations

import unittest

from agents.heuristic_agents import HEURISTIC_POLICIES, evaluate_heuristic_policy
from agents.train_rl import _build_env, evaluate_trained_model
from analysis.trajectory_report import _run_episode
from analysis.evaluator import evaluate_heuristics_table, evaluate_policy
from env.exam_env import ExamStrategyEnv
from env.problem import Problem
from env.reward import compute_step_reward, compute_terminal_reward
from env.state import ExamState, ProblemProgress, ProblemStatus, solved_criteria_from_config
from utils.io import load_config


class _DummyModel:
    def predict(self, obs, deterministic: bool = True):
        return [0, 0], None


class _SequenceModel:
    def __init__(self, actions):
        self._actions = list(actions)
        self._index = 0

    def predict(self, obs, deterministic: bool = True):
        if self._index >= len(self._actions):
            action = self._actions[-1]
        else:
            action = self._actions[self._index]
            self._index += 1
        return list(action), None


def _test_config():
    cfg = load_config("configs/default.yaml")
    cfg.setdefault("data", {})
    cfg["data"]["exam_path"] = "data/25_math_calculus.json"
    cfg["data"].pop("exam_paths", None)
    cfg["data"]["student_path"] = "data/someone.json"
    cfg.setdefault("student", {})
    cfg["student"]["fixed_id"] = "Student"
    return cfg


class EvaluationLoopTests(unittest.TestCase):
    def test_problem_progress_supports_objective_choice_confidences(self):
        problem = Problem(
            pid=1,
            difficulty_level="하",
            difficulty=0.1,
            score=2,
            error_rate=0.1,
            problem_type="objective",
            actual_answer=3,
            choice_rate={"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
        )
        progress = ProblemProgress()
        progress.sync_from_scalar(problem, 0.7)
        self.assertEqual(len(progress.choice_confidences), 5)
        self.assertAlmostEqual(progress.choice_confidences[2], 0.7)
        self.assertAlmostEqual(progress.confidence_score, 0.7)
        self.assertAlmostEqual(progress.observable_confidence(problem), 0.7)
        self.assertAlmostEqual(progress.effective_confidence(problem), 0.7)
        self.assertTrue(
            progress.is_solved(
                problem,
                subjective_threshold=0.5,
                objective_threshold=0.5,
                objective_margin=0.0,
            )
        )

    def test_problem_progress_supports_subjective_answer_confidence(self):
        problem = Problem(
            pid=2,
            difficulty_level="중",
            difficulty=0.5,
            score=3,
            error_rate=0.5,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        progress = ProblemProgress()
        progress.sync_from_scalar(problem, 0.4)
        self.assertEqual(progress.choice_confidences, [])
        self.assertAlmostEqual(progress.answer_confidence, 0.4)
        self.assertAlmostEqual(progress.confidence_score, 0.4)
        self.assertAlmostEqual(progress.observable_confidence(problem), 0.4)
        self.assertAlmostEqual(progress.effective_confidence(problem), 0.4)
        self.assertFalse(
            progress.is_solved(
                problem,
                subjective_threshold=0.5,
                objective_threshold=0.5,
                objective_margin=0.0,
            )
        )

    def test_problem_progress_initializes_objective_priors(self):
        problem = Problem(
            pid=3,
            difficulty_level="하",
            difficulty=0.1,
            score=2,
            error_rate=0.1,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.1, "2": 0.6, "3": 0.1, "4": 0.1, "5": 0.1},
        )
        progress = ProblemProgress()
        progress.initialize_for_problem(problem)
        self.assertEqual(len(progress.choice_confidences), 5)
        self.assertEqual(progress.predicted_choice_index(), 1)
        self.assertAlmostEqual(sum(progress.choice_confidences), 1.0)
        self.assertAlmostEqual(progress.answer_confidence, progress.choice_confidences[1])

    def test_problem_progress_initializes_subjective_prior(self):
        problem = Problem(
            pid=4,
            difficulty_level="중",
            difficulty=0.5,
            score=3,
            error_rate=0.5,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        progress = ProblemProgress()
        progress.initialize_for_problem(problem)
        self.assertEqual(progress.choice_confidences, [])
        self.assertAlmostEqual(progress.answer_confidence, 0.0)

    def test_observation_exposes_type_aware_confidence_slots(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        obs, _ = env.reset(seed=0, options={"student_id": "Student"})
        self.assertEqual(obs.shape[0], 2 + (env.num_problems * 11))
        obs_after, _, _, _, _ = env.step([0, 0])
        self.assertEqual(obs_after.shape[0], 2 + (env.num_problems * 11))

    def test_observation_encodes_subjective_and_objective_confidence_differently(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        obs, _ = env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})

        objective_problem_idx = next(i for i, problem in enumerate(env.problems) if problem.problem_type == "objective")
        subjective_problem_idx = next(i for i, problem in enumerate(env.problems) if problem.problem_type == "subjective")
        per_problem_dim = 11
        base_offset = 2
        confidence_offset = 6

        objective_start = base_offset + (objective_problem_idx * per_problem_dim)
        subjective_start = base_offset + (subjective_problem_idx * per_problem_dim)

        objective_confidences = obs[objective_start + confidence_offset : objective_start + confidence_offset + 5]
        subjective_confidences = obs[subjective_start + confidence_offset : subjective_start + confidence_offset + 5]

        self.assertAlmostEqual(float(sum(objective_confidences)), 1.0, places=5)
        self.assertGreater(float(max(objective_confidences)), 0.0)
        self.assertGreater(float(subjective_confidences[0]), 0.0)
        self.assertTrue(all(float(x) == 0.0 for x in subjective_confidences[1:]))

    def test_solve_more_is_forced_to_next_when_objective_confidence_is_saturated(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student"})
        assert env.state is not None
        current_idx = env.state.current_problem_idx
        problem = env.problems[current_idx]
        if problem.problem_type != "objective":
            objective_idx = next(i for i, p in enumerate(env.problems) if p.problem_type == "objective")
            env.state.current_problem_idx = objective_idx
            env.state.progress[current_idx].status = ProblemStatus.NOT_VISITED
            current_idx = objective_idx
            problem = env.problems[current_idx]
            env.state.progress[current_idx].status = ProblemStatus.IN_PROGRESS
            env.state.visit_order = [current_idx]
        env.state.progress[current_idx].sync_from_scalar(problem, 0.96)
        env.state.same_problem_streak = 0

        _, _, _, _, info = env.step([0, 0])
        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["forced_switch_reason"], "objective_conf_saturated")
        self.assertNotEqual(info["current_problem_idx"], current_idx)

    def test_solve_more_is_forced_to_next_when_streak_limit_is_reached(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student"})
        assert env.state is not None
        current_idx = env.state.current_problem_idx
        env.state.same_problem_streak = int(cfg["exam"]["solve_more_constraints"]["streak_threshold"])

        _, _, _, _, info = env.step([0, 0])
        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["forced_switch_reason"], "streak_limit")
        self.assertNotEqual(info["current_problem_idx"], current_idx)

    def test_solve_more_is_forced_to_next_when_difficulty_time_budget_is_reached(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student"})
        assert env.state is not None
        easy_idx = next(i for i, p in enumerate(env.problems) if p.difficulty_level == "하")
        env.state.current_problem_idx = easy_idx
        env.state.progress[easy_idx].status = ProblemStatus.IN_PROGRESS
        env.state.progress[easy_idx].time_spent_sec = float(cfg["exam"]["difficulty_time_priors_sec"]["하"])
        env.state.visit_order = [easy_idx]

        _, _, _, _, info = env.step([0, 0])
        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["forced_switch_reason"], "difficulty_time_budget_reached")
        self.assertNotEqual(info["current_problem_idx"], easy_idx)

    def test_next_redirects_to_unvisited_problem_during_first_pass(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        assert env.state is not None
        env.state.current_problem_idx = 0
        env.state.progress[0].status = ProblemStatus.IN_PROGRESS
        env.state.progress[1].status = ProblemStatus.NOT_VISITED
        env.state.progress[2].status = ProblemStatus.MOVED_ON
        env.state.visit_order = [0, 2]

        _, _, _, _, info = env.step([1, 2])
        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["current_problem_idx"], 1)

    def test_next_reward_gives_sequential_first_pass_bonus(self):
        problem_a = Problem(pid=1, difficulty_level="하", difficulty=0.1, score=2, error_rate=0.1, problem_type="objective", actual_answer=1, choice_rate={"1": 1.0})
        problem_b = Problem(pid=2, difficulty_level="하", difficulty=0.1, score=2, error_rate=0.1, problem_type="objective", actual_answer=1, choice_rate={"1": 1.0})
        problem_c = Problem(pid=3, difficulty_level="하", difficulty=0.1, score=2, error_rate=0.1, problem_type="objective", actual_answer=1, choice_rate={"1": 1.0})
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[
                ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.8], answer_confidence=0.8),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, choice_confidences=[0.3], answer_confidence=0.3),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, choice_confidences=[0.3], answer_confidence=0.3),
            ],
            total_score=1.4,
            visit_order=[0],
        )
        next_state = ExamState(
            remaining_time_sec=90.0,
            current_problem_idx=1,
            progress=[
                ProblemProgress(status=ProblemStatus.MOVED_ON, choice_confidences=[0.8], answer_confidence=0.8),
                ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.3], answer_confidence=0.3),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, choice_confidences=[0.3], answer_confidence=0.3),
            ],
            total_score=1.4,
            visit_order=[0, 1],
        )
        reward_cfg = {
            "next": {
                "penalty": 0.0,
                "new_problem_bonus": 0.0,
                "coverage_bonus_scale": 0.0,
                "first_pass": {"sequential_bonus": 0.12, "revisit_penalty": -0.05},
                "difficulty_exit": {
                    "easy": {
                        "low_conf_threshold": 0.5,
                        "low_conf_penalty": -0.08,
                        "ready_threshold": 0.7,
                        "ready_bonus": 0.08,
                    }
                },
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem_a, problem_b, problem_c], action_name="next", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, 0.20)

    def test_next_reward_gives_hard_defer_bonus_during_first_pass(self):
        problem = Problem(
            pid=31,
            difficulty_level="최상",
            difficulty=0.9,
            score=4,
            error_rate=0.8,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[
                ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.30),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, answer_confidence=0.0),
            ],
            total_score=1.2,
            visit_order=[0],
        )
        next_state = ExamState(
            remaining_time_sec=90.0,
            current_problem_idx=1,
            progress=[
                ProblemProgress(status=ProblemStatus.MOVED_ON, answer_confidence=0.30),
                ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.0),
            ],
            total_score=1.2,
            visit_order=[0, 1],
        )
        reward_cfg = {
            "next": {
                "penalty": 0.0,
                "new_problem_bonus": 0.0,
                "coverage_bonus_scale": 0.0,
                "first_pass": {"sequential_bonus": 0.0, "revisit_penalty": 0.0},
                "difficulty_exit": {"hard": {"defer_threshold": 0.45, "defer_bonus": 0.04, "ready_threshold": 0.35, "ready_bonus": 0.06}},
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem, problem], action_name="next", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, 0.04)

    def test_forced_switch_prefers_unsolved_hard_revisit_after_first_pass(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        assert env.state is not None
        hard_idx = next(i for i, p in enumerate(env.problems) if p.difficulty_level in {"상", "최상"})
        easy_idx = next(i for i, p in enumerate(env.problems) if p.difficulty_level == "하" and i != hard_idx)
        current_idx = easy_idx
        env.state.current_problem_idx = current_idx
        env.state.visit_order = list(range(env.num_problems))
        for idx, progress in enumerate(env.state.progress):
            progress.status = ProblemStatus.MOVED_ON if idx != current_idx else ProblemStatus.IN_PROGRESS
            progress.time_spent_sec = 30.0
            problem = env.problems[idx]
            progress.sync_from_scalar(problem, 0.96)
        hard_problem = env.problems[hard_idx]
        env.state.progress[hard_idx].sync_from_scalar(hard_problem, 0.3)
        env.state.same_problem_streak = int(cfg["exam"]["solve_more_constraints"]["streak_threshold"])

        _, _, _, _, info = env.step([0, 0])
        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["forced_switch_reason"], "streak_limit")
        self.assertEqual(info["current_problem_idx"], hard_idx)

    def test_priority_revisit_prefers_worked_hard_problem_over_unworked_hard_problem(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        assert env.state is not None

        worked_idx = next(
            i for i, p in enumerate(env.problems)
            if p.difficulty_level == "최상" and p.problem_type == "subjective"
        )
        unworked_idx = next(
            i for i, p in enumerate(env.problems)
            if i != worked_idx and p.difficulty_level in {"상", "최상"} and p.problem_type == "subjective"
        )
        current_idx = next(i for i in range(env.num_problems) if i not in {worked_idx, unworked_idx})

        env.state.current_problem_idx = current_idx
        env.state.visit_order = list(range(env.num_problems))
        for idx, progress in enumerate(env.state.progress):
            problem = env.problems[idx]
            progress.status = ProblemStatus.MOVED_ON if idx != current_idx else ProblemStatus.IN_PROGRESS
            if idx == worked_idx:
                progress.time_spent_sec = 600.0
                progress.answer_confidence = 0.49
                progress.choice_confidences = []
            elif idx == unworked_idx:
                progress.time_spent_sec = 0.0
                progress.answer_confidence = 0.0
                progress.choice_confidences = []
            else:
                progress.time_spent_sec = 30.0
                progress.sync_from_scalar(problem, 0.9)

        target_idx = env._priority_revisit_target(current_idx)
        self.assertEqual(target_idx, worked_idx)

    def test_priority_revisit_respects_recent_entry_cooldown(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        assert env.state is not None

        hard_candidates = [
            i for i, p in enumerate(env.problems)
            if p.difficulty_level in {"상", "최상"} and p.problem_type == "subjective"
        ]
        current_idx, recent_idx, alternate_idx = hard_candidates[:3]
        env.state.current_problem_idx = current_idx
        env.state.visit_order = list(range(env.num_problems))
        for idx, progress in enumerate(env.state.progress):
            problem = env.problems[idx]
            progress.status = ProblemStatus.MOVED_ON if idx != current_idx else ProblemStatus.IN_PROGRESS
            if idx in {recent_idx, alternate_idx}:
                progress.time_spent_sec = 300.0
                progress.answer_confidence = 0.3
                progress.choice_confidences = []
            else:
                progress.time_spent_sec = 30.0
                progress.sync_from_scalar(problem, 0.9)

        env._recent_problem_entries = [recent_idx]
        target_idx = env._priority_revisit_target(current_idx)
        self.assertNotEqual(target_idx, recent_idx)

    def test_next_applies_no_work_revisit_penalty(self):
        cfg = _test_config()
        cfg["exam"]["randomize_start_problem"] = False
        cfg["exam"]["first_pass"]["enabled"] = False
        cfg["exam"]["first_pass"]["enforce_unvisited_before_revisit"] = False
        cfg["exam"]["solve_more_constraints"]["enabled"] = False
        cfg["reward"]["next"]["penalty"] = 0.0
        cfg["reward"]["next"]["new_problem_bonus"] = 0.0
        cfg["reward"]["next"]["coverage_bonus_scale"] = 0.0
        cfg["reward"]["next"]["first_pass"]["sequential_bonus"] = 0.0
        cfg["reward"]["next"]["first_pass"]["revisit_penalty"] = 0.0
        cfg["reward"]["next"]["difficulty_exit"] = {}
        cfg["reward"]["next"]["no_work_revisit_penalty"] = -0.2

        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        assert env.state is not None
        start_idx = env.state.current_problem_idx
        other_idx = (start_idx + 1) % env.num_problems

        _, _, _, _, _ = env.step([0, 0])
        _, reward_b, _, _, _ = env.step([1, other_idx])
        _, reward_c, _, _, _ = env.step([1, start_idx])
        _, reward_d, _, _, info = env.step([1, other_idx])

        self.assertTrue(info["no_work_revisit"])
        self.assertLess(reward_d, reward_c - 0.01)
        self.assertLess(reward_d, reward_b)

    def test_objective_observable_confidence_can_differ_from_correct_confidence(self):
        problem = Problem(
            pid=5,
            difficulty_level="하",
            difficulty=0.1,
            score=2,
            error_rate=0.1,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.6, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1},
        )
        progress = ProblemProgress(choice_confidences=[0.7, 0.2, 0.05, 0.03, 0.02], answer_confidence=0.2)
        self.assertAlmostEqual(progress.observable_confidence(problem), 0.7)
        self.assertAlmostEqual(progress.effective_confidence(problem), 0.2)

    def test_next_reward_shaping_uses_observable_confidence_for_objective(self):
        problem = Problem(
            pid=6,
            difficulty_level="하",
            difficulty=0.1,
            score=2,
            error_rate=0.1,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.6, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1},
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.7, 0.2, 0.05, 0.03, 0.02], answer_confidence=0.2)],
            total_score=0.4,
            visit_order=[0],
        )
        next_state = ExamState(
            remaining_time_sec=90.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.MOVED_ON, choice_confidences=[0.7, 0.2, 0.05, 0.03, 0.02], answer_confidence=0.2)],
            total_score=0.4,
            visit_order=[0],
        )
        reward_cfg = {
            "next": {
                "penalty": 0.0,
                "new_problem_bonus": 0.0,
                "coverage_bonus_scale": 0.0,
                "difficulty_exit": {
                    "easy": {
                        "low_conf_threshold": 0.5,
                        "low_conf_penalty": -0.08,
                        "ready_threshold": 0.7,
                        "ready_bonus": 0.08,
                    }
                },
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem], action_name="next", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, 0.08)

    def test_solve_more_low_marginal_penalty_uses_subjective_confidence_gain(self):
        problem = Problem(
            pid=8,
            difficulty_level="중",
            difficulty=0.5,
            score=3,
            error_rate=0.4,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.40)],
            total_score=1.20,
            same_problem_streak=1,
        )
        next_state = ExamState(
            remaining_time_sec=70.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.405)],
            total_score=1.215,
            same_problem_streak=2,
        )
        reward_cfg = {
            "solve_more": {
                "penalty": 0.0,
                "low_marginal_gain": {
                    "subjective_threshold": 0.01,
                    "objective_threshold": 0.01,
                    "penalty": -0.2,
                },
                "streak": {"threshold": 99, "penalty": 0.0},
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem], action_name="solve_more", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, -0.185)

    def test_solve_more_low_marginal_penalty_uses_correct_choice_gain_for_objective(self):
        problem = Problem(
            pid=9,
            difficulty_level="중",
            difficulty=0.5,
            score=2,
            error_rate=0.4,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.6, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1},
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.7, 0.2, 0.05, 0.03, 0.02], answer_confidence=0.2)],
            total_score=0.4,
            same_problem_streak=1,
        )
        next_state = ExamState(
            remaining_time_sec=70.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.65, 0.205, 0.06, 0.045, 0.04], answer_confidence=0.205)],
            total_score=0.41,
            same_problem_streak=2,
        )
        reward_cfg = {
            "solve_more": {
                "penalty": 0.0,
                "low_marginal_gain": {
                    "subjective_threshold": 0.01,
                    "objective_threshold": 0.01,
                    "penalty": -0.3,
                },
                "streak": {"threshold": 99, "penalty": 0.0},
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem], action_name="solve_more", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, -0.29)

    def test_solve_more_applies_saturation_penalty_for_objective(self):
        problem = Problem(
            pid=12,
            difficulty_level="하",
            difficulty=0.1,
            score=3,
            error_rate=0.1,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.04, 0.88, 0.03, 0.03, 0.02], answer_confidence=0.88)],
            total_score=2.64,
            same_problem_streak=1,
        )
        next_state = ExamState(
            remaining_time_sec=70.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, choice_confidences=[0.01, 0.93, 0.02, 0.02, 0.02], answer_confidence=0.93)],
            total_score=2.79,
            same_problem_streak=2,
        )
        reward_cfg = {
            "solve_more": {
                "penalty": 0.0,
                "low_marginal_gain": {"subjective_threshold": 0.0, "objective_threshold": 0.0, "penalty": 0.0},
                "saturation": {
                    "objective": {"threshold": 0.90, "penalty": -0.05},
                    "subjective": {"threshold": 0.90, "penalty": -0.03},
                },
                "streak": {"threshold": 99, "penalty": 0.0, "extra_penalty_scale": 0.0, "max_extra_steps": 99},
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem], action_name="solve_more", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, 0.10)

    def test_solve_more_streak_penalty_grows_after_threshold(self):
        problem = Problem(
            pid=13,
            difficulty_level="중",
            difficulty=0.5,
            score=3,
            error_rate=0.4,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        prev_state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.20)],
            total_score=0.60,
            same_problem_streak=10,
        )
        next_state = ExamState(
            remaining_time_sec=70.0,
            current_problem_idx=0,
            progress=[ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.25)],
            total_score=0.75,
            same_problem_streak=11,
        )
        reward_cfg = {
            "solve_more": {
                "penalty": 0.0,
                "low_marginal_gain": {"subjective_threshold": 0.0, "objective_threshold": 0.0, "penalty": 0.0},
                "saturation": {
                    "objective": {"threshold": 0.90, "penalty": 0.0},
                    "subjective": {"threshold": 0.90, "penalty": 0.0},
                },
                "streak": {"threshold": 8, "penalty": 0.0, "extra_penalty_scale": -0.01, "max_extra_steps": 40},
            }
        }
        reward = compute_step_reward(prev_state, next_state, [problem], action_name="solve_more", reward_cfg=reward_cfg)
        self.assertAlmostEqual(reward, 0.12)

    def test_terminal_reward_penalizes_high_time_concentration(self):
        state = ExamState(
            remaining_time_sec=0.0,
            current_problem_idx=0,
            progress=[
                ProblemProgress(status=ProblemStatus.MOVED_ON, time_spent_sec=95.0),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, time_spent_sec=3.0),
                ProblemProgress(status=ProblemStatus.NOT_VISITED, time_spent_sec=2.0),
            ],
        )
        reward_cfg = {
            "terminal": {
                "timeout_penalty": -1.0,
                "completion_bonus": 0.0,
                "concentration": {
                    "top1": {"threshold": 0.70, "penalty_scale": -8.0},
                    "top2": {"threshold": 0.90, "penalty_scale": -6.0},
                },
            }
        }
        reward = compute_terminal_reward(state, [], reward_cfg, timed_out=True, step_limited=False)
        self.assertAlmostEqual(reward, -3.48)

    def test_objective_solved_requires_margin_when_configured(self):
        problem = Problem(
            pid=7,
            difficulty_level="중",
            difficulty=0.5,
            score=3,
            error_rate=0.4,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.3, "2": 0.3, "3": 0.2, "4": 0.1, "5": 0.1},
        )
        progress = ProblemProgress(choice_confidences=[0.44, 0.48, 0.04, 0.02, 0.02], answer_confidence=0.48)
        self.assertFalse(
            progress.is_solved(
                problem,
                subjective_threshold=0.5,
                objective_threshold=0.5,
                objective_margin=0.05,
            )
        )
        progress = ProblemProgress(choice_confidences=[0.10, 0.62, 0.10, 0.09, 0.09], answer_confidence=0.62)
        self.assertTrue(
            progress.is_solved(
                problem,
                subjective_threshold=0.5,
                objective_threshold=0.5,
                objective_margin=0.05,
            )
        )

    def test_solved_criteria_loaded_from_config(self):
        cfg = _test_config()
        criteria = solved_criteria_from_config(cfg)
        self.assertEqual(criteria["subjective_threshold"], 0.5)
        self.assertEqual(criteria["objective_threshold"], 0.5)
        self.assertEqual(criteria["objective_margin"], 0.05)

    def test_state_type_specific_metrics(self):
        subjective = Problem(
            pid=10,
            difficulty_level="중",
            difficulty=0.4,
            score=3,
            error_rate=0.4,
            problem_type="subjective",
            actual_answer=None,
            choice_rate=None,
        )
        objective = Problem(
            pid=11,
            difficulty_level="하",
            difficulty=0.2,
            score=2,
            error_rate=0.2,
            problem_type="objective",
            actual_answer=2,
            choice_rate={"1": 0.5, "2": 0.2, "3": 0.1, "4": 0.1, "5": 0.1},
        )
        state = ExamState(
            remaining_time_sec=100.0,
            current_problem_idx=0,
            progress=[
                ProblemProgress(status=ProblemStatus.IN_PROGRESS, answer_confidence=0.6),
                ProblemProgress(status=ProblemStatus.MOVED_ON, choice_confidences=[0.2, 0.55, 0.1, 0.1, 0.05], answer_confidence=0.55),
            ],
        )
        criteria = {"subjective_threshold": 0.5, "objective_threshold": 0.5, "objective_margin": 0.05}
        problems = [subjective, objective]
        self.assertAlmostEqual(state.mean_subjective_confidence(problems), 0.6)
        self.assertAlmostEqual(state.objective_dominance_rate(problems), 1.0)
        self.assertAlmostEqual(state.subjective_solved_rate(problems, **criteria), 1.0)
        self.assertAlmostEqual(state.objective_solved_rate(problems, **criteria), 1.0)

    def test_evaluate_trained_model_runs_requested_episode_count(self):
        cfg = _test_config()
        metrics = evaluate_trained_model(
            model=_DummyModel(),
            config=cfg,
            n_episodes=3,
            algorithm="ppo",
            seed=123,
        )
        self.assertEqual(metrics["episodes"], 3.0)
        self.assertGreaterEqual(metrics["mean_score"], 0.0)
        self.assertIn("mean_objective_dominance_rate", metrics)
        self.assertIn("mean_subjective_confidence", metrics)
        self.assertIn("mean_subjective_solved_rate", metrics)
        self.assertIn("mean_objective_solved_rate", metrics)

    def test_evaluate_policy_returns_all_episode_records(self):
        cfg = _test_config()
        result = evaluate_policy(
            config=cfg,
            policy_name="easy_first",
            episodes=4,
            student_id="Student",
            seed=7,
        )
        self.assertEqual(len(result["episode_records"]), 4)
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["mode"], "heuristic")
        self.assertIn("mean_objective_dominance_rate", result["summary"])
        self.assertIn("mean_subjective_confidence", result["summary"])

    def test_evaluate_heuristics_table_includes_all_heuristics(self):
        cfg = _test_config()
        rows = evaluate_heuristics_table(
            config=cfg,
            episodes=2,
            student_id="Student",
            seed=11,
        )
        self.assertEqual(len(rows), len(HEURISTIC_POLICIES))

    def test_evaluate_heuristic_policy_runs_multiple_episodes(self):
        cfg = _test_config()

        def _env_factory():
            return ExamStrategyEnv(config=cfg, random_seed=99)

        result = evaluate_heuristic_policy(
            env_factory=_env_factory,
            policy_name="index_order",
            episodes=3,
            seed=5,
        )
        self.assertEqual(result["episodes"], 3)
        self.assertGreater(result["mean_steps"], 0.0)

    def test_evaluate_trained_model_is_deterministic_for_fixed_seed(self):
        cfg = _test_config()
        first = evaluate_trained_model(
            model=_DummyModel(),
            config=cfg,
            n_episodes=3,
            algorithm="ppo",
            seed=123,
        )
        second = evaluate_trained_model(
            model=_DummyModel(),
            config=cfg,
            n_episodes=3,
            algorithm="ppo",
            seed=123,
        )
        self.assertEqual(first, second)

    def test_fixed_order_free_time_wrapper_forces_next_problem_order(self):
        cfg = _test_config()
        cfg["exam"]["randomize_start_problem"] = False
        cfg["exam"]["solve_more_constraints"]["enabled"] = False
        cfg["training"]["strategy_constraint"] = {"name": "fixed_order_free_time", "min_time_per_problem_sec": 0}
        env = _build_env(config=cfg, seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})

        _, _, _, _, info = env.step(1)

        self.assertEqual(info["action_name"], "next")
        self.assertEqual(info["current_problem_idx"], 1)
        self.assertEqual(info["visit_order"], [1, 2])

    def test_fixed_order_free_time_wrapper_reserves_time_for_future_problems(self):
        cfg = _test_config()
        cfg["exam"]["randomize_start_problem"] = False
        cfg["exam"]["total_time_sec"] = 40
        cfg["exam"]["action_time_unit_sec"] = 10
        cfg["exam"]["switch_time_sec"] = 0
        cfg["exam"]["solve_more_constraints"]["enabled"] = False
        cfg["training"]["strategy_constraint"] = {"name": "fixed_order_free_time", "min_time_per_problem_sec": 10}
        env = _build_env(config=cfg, seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})

        infos = []
        for _ in range(7):
            _, _, terminated, truncated, info = env.step(0)
            infos.append(info)
            if terminated or truncated:
                break

        self.assertEqual([1, 2, 3, 4], infos[-1]["visit_order"])
        self.assertEqual(infos[-1]["current_problem_idx"], 3)

    def test_equal_time_free_order_wrapper_forces_work_until_budget(self):
        cfg = _test_config()
        cfg["exam"]["randomize_start_problem"] = False
        cfg["exam"]["action_time_unit_sec"] = 10
        cfg["exam"]["switch_time_sec"] = 0
        cfg["exam"]["solve_more_constraints"]["enabled"] = False
        cfg["training"]["strategy_constraint"] = {"name": "equal_time_free_order", "time_budget_sec": 20}
        env = _build_env(config=cfg, seed=0)
        env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})

        _, _, _, _, first_info = env.step(5)
        _, _, _, _, second_info = env.step(5)
        _, _, _, _, third_info = env.step(5)

        self.assertEqual(first_info["action_name"], "solve_more")
        self.assertEqual(second_info["action_name"], "solve_more")
        self.assertEqual(third_info["action_name"], "next")
        self.assertEqual(third_info["current_problem_idx"], 5)

    def test_evaluate_policy_is_deterministic_for_fixed_seed(self):
        cfg = _test_config()
        first = evaluate_policy(
            config=cfg,
            policy_name="index_order",
            episodes=3,
            student_id="Student",
            seed=123,
        )
        second = evaluate_policy(
            config=cfg,
            policy_name="index_order",
            episodes=3,
            student_id="Student",
            seed=123,
        )
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["problem_avg_time"], second["problem_avg_time"])
        self.assertEqual(first["episode_records"], second["episode_records"])

    def test_trajectory_report_exposes_type_aware_problem_details(self):
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        solved_criteria = solved_criteria_from_config(cfg)
        report = _run_episode(
            env,
            rl_model=None,
            policy_name="index_order",
            seed=0,
            reset_options={"student_id": "Student"},
            max_logged_steps=5,
            solved_criteria=solved_criteria,
        )
        self.assertIn("type_breakdown", report)
        self.assertIn("objective_top5_by_time", report["type_breakdown"])
        self.assertIn("subjective_top5_by_time", report["type_breakdown"])
        self.assertIn("problem_type", report["problem_ranking"][0])
        self.assertIn("observable_confidence", report["problem_ranking"][0])
        self.assertIn("effective_confidence", report["problem_ranking"][0])
        self.assertIn("is_solved", report["problem_ranking"][0])
        self.assertIn("visit_count", report["problem_ranking"][0])
        self.assertIn("was_revisited", report["problem_ranking"][0])
        self.assertIn("revisit_count", report)
        self.assertIn("revisited_problem_indices", report)
        self.assertIn("revisited_hard_problem_events", report)
        self.assertIn("prev_problem_type", report["trajectory_head"][0])
        self.assertIn("prev_observable_confidence", report["trajectory_head"][0])
        self.assertIn("prev_effective_confidence", report["trajectory_head"][0])

    def test_trajectory_report_counts_only_meaningful_revisits(self):
        cfg = _test_config()
        cfg["exam"]["randomize_start_problem"] = False
        cfg["exam"]["total_time_sec"] = 130
        cfg["exam"]["first_pass"]["enabled"] = False
        cfg["exam"]["first_pass"]["enforce_unvisited_before_revisit"] = False
        cfg["exam"]["solve_more_constraints"]["enabled"] = False

        probe_env = ExamStrategyEnv(config=cfg, random_seed=0)
        _, probe_info = probe_env.reset(seed=0, options={"student_id": "Student", "exam_index": 0})
        start_idx = int(probe_info["start_problem_idx"])
        other_idx = (start_idx + 1) % probe_env.num_problems

        model = _SequenceModel(
            [
                [0, 0],  # work on the start problem
                [1, other_idx],
                [1, start_idx],  # re-enter start problem but do no work
                [1, other_idx],
                [0, 0],  # work on the other problem
                [1, start_idx],
                [0, 0],  # meaningful revisit on the start problem
            ]
        )

        env = ExamStrategyEnv(config=cfg, random_seed=0)
        solved_criteria = solved_criteria_from_config(cfg)
        report = _run_episode(
            env,
            rl_model=model,
            policy_name=None,
            seed=0,
            reset_options={"student_id": "Student", "exam_index": 0},
            max_logged_steps=10,
            solved_criteria=solved_criteria,
        )

        self.assertEqual(report["revisit_count"], 1)
        self.assertEqual(report["revisited_problem_indices"], [start_idx + 1])
        start_snapshot = next(item for item in report["problem_ranking"] if item["problem_idx"] == start_idx + 1)
        other_snapshot = next(item for item in report["problem_ranking"] if item["problem_idx"] == other_idx + 1)
        self.assertEqual(start_snapshot["entry_count"], 3)
        self.assertEqual(start_snapshot["visit_count"], 2)
        self.assertEqual(other_snapshot["entry_count"], 2)
        self.assertEqual(other_snapshot["visit_count"], 1)


    # ── Phase 1/4/5 regression tests ─────────────────────────────────────────

    def test_correct_rate_is_loaded_from_json(self):
        from env.problem import load_problem_list
        problems = load_problem_list("data/25_math_calculus.json")
        # All problems in the JSON have correct_rate; none should be None
        for p in problems:
            self.assertIsNotNone(p.correct_rate, msg=f"pid={p.pid} missing correct_rate")
            self.assertGreaterEqual(float(p.correct_rate), 0.0)
            self.assertLessEqual(float(p.correct_rate), 1.0)

    def test_old_start_mode_is_default(self):
        # Default config must not use allow_agent_selected_start_problem
        cfg = _test_config()
        self.assertFalse(cfg.get("exam", {}).get("allow_agent_selected_start_problem", False))
        env = ExamStrategyEnv(config=cfg, random_seed=7)
        _, info = env.reset(seed=7)
        self.assertGreaterEqual(info["start_problem_idx"], 0)
        assert env.state is not None
        self.assertGreaterEqual(env.state.current_problem_idx, 0)
        # At least one problem must be IN_PROGRESS
        self.assertTrue(
            any(p.status.value == "IN_PROGRESS" for p in env.state.progress),
            "Old mode must start with exactly one problem IN_PROGRESS",
        )

    def test_new_start_mode_resets_with_sentinel(self):
        cfg = _test_config()
        cfg["exam"]["allow_agent_selected_start_problem"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=7)
        _, info = env.reset(seed=7)
        self.assertEqual(info["start_problem_idx"], -1)
        assert env.state is not None
        self.assertEqual(env.state.current_problem_idx, -1)
        self.assertEqual(env.state.visit_order, [])
        self.assertTrue(
            all(p.status.value == "NOT_VISITED" for p in env.state.progress),
            "New mode: all problems must be NOT_VISITED at reset",
        )

    def test_shuffle_problem_order_on_reset_is_seeded(self):
        cfg = load_config("configs/ppo/train_mid.yaml")
        cfg.setdefault("data", {})
        cfg["data"]["exam_path"] = "data/25_math_calculus.json"
        cfg["data"].pop("exam_paths", None)
        cfg["exam"]["shuffle_problem_order_on_reset"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=7)
        _, info_a = env.reset(seed=7)
        _, info_b = env.reset(seed=7)
        _, info_c = env.reset(seed=8)

        self.assertEqual(info_a["problem_pids"], info_b["problem_pids"])
        self.assertNotEqual(info_a["problem_pids"], info_c["problem_pids"])
        self.assertEqual(sorted(info_a["problem_pids"]), sorted(info_c["problem_pids"]))

    def test_new_start_mode_first_action_is_select_start(self):
        cfg = _test_config()
        cfg["exam"]["allow_agent_selected_start_problem"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=7)
        env.reset(seed=7)
        target = 3
        _, reward, _, _, info = env.step([0, target])  # action_type ignored
        self.assertEqual(info["action_name"], "select_start")
        self.assertEqual(info["target_problem_idx"], target)
        self.assertAlmostEqual(reward, 0.0, places=9)
        assert env.state is not None
        self.assertEqual(env.state.current_problem_idx, target)

    def test_new_start_mode_second_action_is_normal(self):
        cfg = _test_config()
        cfg["exam"]["allow_agent_selected_start_problem"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=7)
        env.reset(seed=7)
        env.step([1, 2])  # select_start
        _, _, _, _, info = env.step([0, 0])  # normal solve_more
        self.assertEqual(info["action_name"], "solve_more")

    def test_marginal_gain_greedy_is_in_heuristic_policies(self):
        from agents.heuristic_agents import HEURISTIC_POLICIES
        self.assertIn("marginal_gain_greedy", HEURISTIC_POLICIES)

    def test_marginal_gain_greedy_runs_episode(self):
        from agents.heuristic_agents import run_heuristic_episode
        cfg = _test_config()
        env = ExamStrategyEnv(config=cfg, random_seed=42)
        stats = run_heuristic_episode(env, "marginal_gain_greedy", reset_seed=42)
        self.assertGreater(stats.total_score, 0.0)
        self.assertGreater(stats.visited_count, 0)
        self.assertTrue(stats.done)

    def test_marginal_gain_greedy_handles_new_start_mode(self):
        from agents.heuristic_agents import run_heuristic_episode
        cfg = _test_config()
        cfg["exam"]["allow_agent_selected_start_problem"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=42)
        stats = run_heuristic_episode(env, "marginal_gain_greedy", reset_seed=42)
        self.assertGreater(stats.total_score, 0.0)
        self.assertTrue(stats.done)

    def test_evaluate_heuristics_table_includes_marginal_gain_greedy(self):
        cfg = _test_config()
        rows = evaluate_heuristics_table(config=cfg, episodes=1, student_id="Student", seed=0)
        policy_names = [row["policy_name"] for row in rows]
        self.assertIn("marginal_gain_greedy", policy_names)

    def test_encode_select_start_action(self):
        cfg = _test_config()
        cfg["exam"]["allow_agent_selected_start_problem"] = True
        env = ExamStrategyEnv(config=cfg, random_seed=0)
        env.reset(seed=0)
        action = env.encode_select_start_action(5)
        self.assertEqual(int(action[1]), 5)


if __name__ == "__main__":
    unittest.main()
