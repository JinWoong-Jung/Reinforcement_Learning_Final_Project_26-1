from __future__ import annotations

import unittest

from env.dynamics import confidence_curve, confidence_params, guessing_prob
from env.problem import Problem, choice_entropy
from env.student import StudentProfile


def _make_student(theta: float = 0.0) -> StudentProfile:
    return StudentProfile(
        student_id="test",
        theta=theta,
    )


def _make_objective(correct_rate: float | None, difficulty: float, choice_rate: dict | None = None) -> Problem:
    if choice_rate is None:
        # Default 5 equal choices
        choice_rate = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}
    return Problem(
        pid=0,
        difficulty_level="중",
        difficulty=difficulty,
        score=3,
        error_rate=0.5,
        problem_type="objective",
        actual_answer=1,
        choice_rate=choice_rate,
        correct_rate=correct_rate,
    )


def _make_subjective(correct_rate: float | None, difficulty: float) -> Problem:
    return Problem(
        pid=0,
        difficulty_level="중",
        difficulty=difficulty,
        score=4,
        error_rate=0.5,
        problem_type="subjective",
        correct_rate=correct_rate,
    )


TIME_STEPS = [0, 10, 30, 60, 90, 180, 300, 600]


class DynamicsMonotonicityTests(unittest.TestCase):
    """Confidence must never decrease as time increases."""

    def _assert_monotone(self, problem: Problem, student: StudentProfile, dynamics_cfg: dict | None = None) -> None:
        confs = [confidence_curve(problem, student, t, dynamics_cfg) for t in TIME_STEPS]
        for i in range(1, len(confs)):
            self.assertGreaterEqual(
                confs[i],
                confs[i - 1] - 1e-9,
                msg=f"Confidence decreased from t={TIME_STEPS[i-1]} ({confs[i-1]:.6f}) to t={TIME_STEPS[i]} ({confs[i]:.6f})",
            )

    def test_monotone_objective_easy(self):
        self._assert_monotone(_make_objective(correct_rate=0.90, difficulty=0.1), _make_student(0.5))

    def test_monotone_objective_hard(self):
        self._assert_monotone(_make_objective(correct_rate=0.10, difficulty=0.9), _make_student(0.5))

    def test_monotone_subjective_mid(self):
        self._assert_monotone(_make_subjective(correct_rate=0.50, difficulty=0.5), _make_student(0.5))

    def test_monotone_low_ability_student(self):
        self._assert_monotone(_make_objective(correct_rate=0.50, difficulty=0.5), _make_student(-1.0))

    def test_monotone_high_ability_student(self):
        self._assert_monotone(_make_objective(correct_rate=0.50, difficulty=0.5), _make_student(1.0))

    def test_monotone_with_custom_dynamics_cfg(self):
        cfg = {"alpha": 3.0, "tau": 30.0, "beta": 2.5, "ambiguity_weight": 1.0}
        self._assert_monotone(_make_objective(correct_rate=0.50, difficulty=0.5), _make_student(0.5), cfg)


class DynamicsAnchorSanityTests(unittest.TestCase):
    """Easier problems (lower difficulty) must yield
    higher confidence than harder ones at the same time."""

    def test_easier_difficulty_means_higher_confidence(self):
        student = _make_student(0.5)
        easy = _make_objective(correct_rate=0.15, difficulty=0.1)
        hard = _make_objective(correct_rate=0.85, difficulty=0.9)
        for t in [30, 90, 300]:
            conf_easy = confidence_curve(easy, student, t)
            conf_hard = confidence_curve(hard, student, t)
            self.assertGreater(
                conf_easy, conf_hard,
                msg=f"At t={t}s, easy ({conf_easy:.4f}) should beat hard ({conf_hard:.4f})",
            )

    def test_anchor_source_difficulty_takes_priority_by_default(self):
        student = _make_student(0.5)
        # Both have the same correct_rate, but difficulty differs → difficulty wins
        easy = _make_objective(correct_rate=0.50, difficulty=0.1)
        hard = _make_objective(correct_rate=0.50, difficulty=0.9)
        t = 90
        self.assertGreater(confidence_curve(easy, student, t), confidence_curve(hard, student, t))

    def test_anchor_falls_back_to_difficulty_when_correct_rate_is_none(self):
        student = _make_student(0.5)
        easy = _make_objective(correct_rate=None, difficulty=0.1)
        hard = _make_objective(correct_rate=None, difficulty=0.9)
        t = 90
        self.assertGreater(confidence_curve(easy, student, t), confidence_curve(hard, student, t))

    def test_anchor_source_correct_rate_can_be_requested_explicitly(self):
        student = _make_student(0.5)
        cfg = {"anchor_source": "correct_rate"}
        easy = Problem(
            pid=0, difficulty_level="중", difficulty=0.5, score=3, error_rate=0.4,
            problem_type="objective", actual_answer=1,
            choice_rate={"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
            correct_rate=0.8,
        )
        hard = Problem(
            pid=1, difficulty_level="중", difficulty=0.5, score=3, error_rate=0.4,
            problem_type="objective", actual_answer=1,
            choice_rate={"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2},
            correct_rate=0.2,
        )
        self.assertGreater(
            confidence_curve(easy, student, 90, cfg),
            confidence_curve(hard, student, 90, cfg),
        )

    def test_more_able_student_has_higher_confidence(self):
        problem = _make_objective(correct_rate=0.50, difficulty=0.5)
        low = _make_student(-1.0)
        high = _make_student(1.0)
        for t in [30, 90, 300]:
            self.assertGreater(
                confidence_curve(problem, high, t),
                confidence_curve(problem, low, t),
                msg=f"At t={t}s, high-ability student should have higher confidence",
            )


class DynamicsAmbiguitySanityTests(unittest.TestCase):
    """Higher ambiguity (choice_entropy) must slow confidence rise for objective problems."""

    def _make_high_ambiguity(self) -> Problem:
        # Uniform distribution over 4 choices → max entropy
        return Problem(
            pid=0, difficulty_level="중", difficulty=0.5, score=3, error_rate=0.4,
            problem_type="objective", actual_answer=1,
            choice_rate={"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25},
            correct_rate=0.50,
        )

    def _make_low_ambiguity(self) -> Problem:
        # One dominant choice → low entropy
        return Problem(
            pid=1, difficulty_level="중", difficulty=0.5, score=3, error_rate=0.4,
            problem_type="objective", actual_answer=1,
            choice_rate={"1": 0.95, "2": 0.02, "3": 0.02, "4": 0.01},
            correct_rate=0.50,
        )

    def test_entropy_values_differ_as_expected(self):
        high = self._make_high_ambiguity()
        low = self._make_low_ambiguity()
        self.assertAlmostEqual(choice_entropy(high), 1.0, places=5)
        self.assertLess(choice_entropy(low), 0.3)

    def test_high_ambiguity_gives_lower_confidence_at_same_difficulty(self):
        student = _make_student(0.5)
        high = self._make_high_ambiguity()
        low = self._make_low_ambiguity()
        # With a nonzero ambiguity_weight, high-entropy problem should be harder
        cfg = {"ambiguity_weight": 0.5}
        for t in [60, 120, 300]:
            conf_high = confidence_curve(high, student, t, cfg)
            conf_low = confidence_curve(low, student, t, cfg)
            self.assertLess(
                conf_high, conf_low,
                msg=f"At t={t}s, high-ambiguity ({conf_high:.4f}) should be lower than low-ambiguity ({conf_low:.4f})",
            )

    def test_zero_ambiguity_weight_makes_entropy_irrelevant(self):
        student = _make_student(0.5)
        high = self._make_high_ambiguity()
        low = self._make_low_ambiguity()
        cfg = {"ambiguity_weight": 0.0}
        # With gamma=0, entropy has no effect → confidences must be equal
        for t in [60, 300]:
            self.assertAlmostEqual(
                confidence_curve(high, student, t, cfg),
                confidence_curve(low, student, t, cfg),
                places=6,
            )


class DynamicsFloorTests(unittest.TestCase):
    """Floor probabilities must match expected semantics."""

    def test_objective_floor_is_fixed_to_point_two(self):
        for k in [2, 4, 5]:
            choice_rate = {str(i + 1): 1.0 / k for i in range(k)}
            problem = Problem(
                pid=0, difficulty_level="하", difficulty=0.1, score=2, error_rate=0.1,
                problem_type="objective", actual_answer=1, choice_rate=choice_rate,
            )
            self.assertAlmostEqual(guessing_prob(problem), 0.2, places=6)

    def test_subjective_floor_defaults_to_zero(self):
        problem = _make_subjective(correct_rate=0.5, difficulty=0.5)
        self.assertAlmostEqual(guessing_prob(problem), 0.0, places=6)

    def test_subjective_floor_is_configurable(self):
        problem = _make_subjective(correct_rate=0.5, difficulty=0.5)
        cfg = {"subjective_floor": 0.05}
        self.assertAlmostEqual(guessing_prob(problem, cfg), 0.05, places=6)

    def test_confidence_at_t0_is_at_least_floor(self):
        student = _make_student(-1.0)
        for problem in [
            _make_objective(correct_rate=0.05, difficulty=0.95),
            _make_subjective(correct_rate=0.05, difficulty=0.95),
        ]:
            floor = guessing_prob(problem)
            conf = confidence_curve(problem, student, time_spent=0.0)
            self.assertGreaterEqual(conf, floor - 1e-9)

    def test_confidence_always_in_0_1(self):
        student = _make_student(0.5)
        for problem in [
            _make_objective(correct_rate=0.99, difficulty=0.01),
            _make_objective(correct_rate=0.01, difficulty=0.99),
            _make_subjective(correct_rate=0.99, difficulty=0.01),
        ]:
            for t in [0, 30, 300, 6000]:
                conf = confidence_curve(problem, student, t)
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)


class DynamicsParamsTests(unittest.TestCase):
    """confidence_params returns the expected interpretable tuple."""

    def test_params_tuple_length(self):
        problem = _make_objective(correct_rate=0.5, difficulty=0.5)
        student = _make_student(0.5)
        params = confidence_params(problem, student)
        self.assertEqual(len(params), 6)

    def test_params_floor_matches_guessing_prob(self):
        problem = _make_objective(correct_rate=0.5, difficulty=0.5)
        student = _make_student(0.5)
        floor, *_ = confidence_params(problem, student)
        self.assertAlmostEqual(floor, guessing_prob(problem), places=6)

    def test_higher_ability_gives_higher_theta(self):
        problem = _make_objective(correct_rate=0.5, difficulty=0.5)
        low = _make_student(-1.0)
        high = _make_student(1.0)
        _, theta_low, *_ = confidence_params(problem, low)
        _, theta_high, *_ = confidence_params(problem, high)
        self.assertGreater(theta_high, theta_low)

    def test_profile_theta_is_used_directly(self):
        problem = _make_objective(correct_rate=0.5, difficulty=0.5)
        _, theta, *_ = confidence_params(problem, _make_student(2.5))
        self.assertAlmostEqual(theta, 2.5)
