"""Time-allocation RL environment for exam strategy optimisation.

The agent's sole decision at each step is: *which problem gets the next
`action_time_unit_sec` seconds?*  Problem visit order is irrelevant; the
action space is ``Discrete(num_problems)``.

Design differences from ExamStrategyEnv
-----------------------------------------
* No ``next`` / ``solve_more`` split — every action allocates time.
* No first-pass, revisit policy, or switch-time per action.
  Optional ``exam.reserve_switch_time: true`` deducts the total overhead
  ``(num_problems - 1) * switch_time_sec`` from the budget once at reset.
* Episode ends when the budget is exhausted or ``max_steps`` is hit.
* Observation is 1 + num_problems * 10 (global + per-problem).
* Step reward = Δ expected score; terminal reward from reward.terminal config.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    class _FallbackEnv:
        metadata: dict = {}

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low; self.high = high; self.shape = shape; self.dtype = dtype

    class _FallbackDiscrete:
        def __init__(self, n): self.n = int(n)

    class _FallbackSpaces:
        Box = _FallbackBox
        Discrete = _FallbackDiscrete

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()  # type: ignore[assignment]
    spaces = _FallbackSpaces()  # type: ignore[assignment]

from .dynamics import (
    apply_time_cost,
    confidence_curve,
    confidence_static_params,
    expected_total_score,
)
from .problem import Problem, load_exam_json
from .reward import compute_terminal_reward
from .state import ExamState, ProblemProgress, ProblemStatus, solved_criteria_from_config
from .student import StudentProfile, create_level_profile, load_student_profiles, sample_student_profile


class TimeAllocationEnv(gym.Env):  # type: ignore[misc]
    """Pure time-allocation exam environment.

    Observation (float32, shape ``(1 + num_problems * 11,)``):
        [0]      remaining_time / available_time
        per problem (11 each):
            time_spent / available_time
            difficulty_level  (0.0 하 … 1.0 최상)
            score / max_score
            problem_type      (0.0 objective, 1.0 subjective)
            error_rate
            confidence slots  [c0, c1, c2, c3, c4]  (5 slots)
            marginal_gain     (Δ expected_score from one more action unit)

    Action:
        ``Discrete(num_problems)`` — index of the problem to invest time in.
    """

    metadata = {"render_modes": []}

    DIFFICULTY_LEVEL_MAP: dict[str, float] = {
        "하": 0.0, "중하": 0.2, "중": 0.4,
        "중상": 0.6, "상": 0.8, "최상": 1.0, "unknown": 0.5,
    }
    PROBLEM_TYPE_MAP: dict[str, float] = {"objective": 0.0, "subjective": 1.0}

    def __init__(
        self,
        config: dict[str, Any],
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.reward_cfg = dict(config.get("reward", {}))
        self.exam_cfg = dict(config.get("exam", {}))
        self.data_cfg = dict(config.get("data", {}))
        self.student_cfg = dict(config.get("student", {}))
        self.dynamics_cfg = dict(config.get("dynamics", {}))
        self.max_steps_cfg = self.exam_cfg.get("max_steps")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def _resolve(path: str | None, default_rel: str) -> str:
            if path is None:
                return os.path.join(project_root, default_rel)
            return path if os.path.isabs(path) else os.path.join(project_root, path)

        cfg_paths = list(self.data_cfg.get("exam_paths", []))
        self.exam_data_paths = (
            [_resolve(p, os.path.join("data", "mock_exam.json")) for p in cfg_paths]
            if cfg_paths
            else [_resolve(self.data_cfg.get("exam_path"), os.path.join("data", "mock_exam.json"))]
        )
        self.student_data_path = _resolve(
            self.data_cfg.get("student_path"), os.path.join("data", "mock_students.json")
        )
        self.student_preset_path = _resolve(
            self.data_cfg.get("student_preset_path"),
            os.path.join("data", "student_level_presets.json"),
        )

        self.fixed_student_level: str | None = self.student_cfg.get("fixed_level")
        self.fixed_student_id: str | None = self.student_cfg.get("fixed_id")

        # Load exam bank — all exams must have the same number of problems.
        self.exam_bank: list[dict[str, Any]] = []
        base_n: int | None = None
        for path in self.exam_data_paths:
            exam_data = load_exam_json(path)
            problems = [Problem.from_dict(x) for x in exam_data["problems"]]
            if base_n is None:
                base_n = len(problems)
            elif len(problems) != base_n:
                raise ValueError("All exam files must have the same number of problems.")
            self.exam_bank.append({"path": path, "exam_data": exam_data, "problems": problems})
        if not self.exam_bank:
            raise ValueError("At least one exam file must be provided.")

        self.problems: list[Problem] = list(self.exam_bank[0]["problems"])
        self.num_problems: int = len(self.problems)
        self.total_time_sec: float = float(
            self.exam_cfg.get("total_time_sec",
                              self.exam_bank[0]["exam_data"].get("total_time_sec", 6000))
        )
        self.action_time_unit_sec: float = float(self.exam_cfg.get("action_time_unit_sec", 30))
        self.switch_time_sec: float = float(self.exam_cfg.get("switch_time_sec", 10))
        self.reserve_switch_time: bool = bool(self.exam_cfg.get("reserve_switch_time", False))
        self.shuffle_problem_order_on_reset: bool = bool(
            self.exam_cfg.get("shuffle_problem_order_on_reset", False)
        )

        # max_steps safety cap: enough for full budget + small margin
        _budget = self._compute_available_time(self.total_time_sec, self.num_problems)
        default_max_steps = int(math.ceil(_budget / max(self.action_time_unit_sec, 1.0))) + self.num_problems
        self.max_steps: int = int(self.max_steps_cfg) if self.max_steps_cfg is not None else default_max_steps

        self.rng = np.random.default_rng(random_seed)
        self.student_profiles = load_student_profiles(self.student_data_path)
        self.current_student: StudentProfile | None = None
        self._mg_params_cache: list[tuple[float, float, float, float, float]] | None = None
        self.state: ExamState | None = None
        self.problem_order: list[int] = list(range(self.num_problems))
        self.solved_criteria = solved_criteria_from_config(config)
        self.exam_data_path: str = self.exam_data_paths[0]

        self.action_space = spaces.Discrete(self.num_problems)
        obs_dim = 1 + self.num_problems * 11
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_available_time(self, total_time: float, num_problems: int) -> float:
        if self.reserve_switch_time:
            overhead = max(num_problems - 1, 0) * self.switch_time_sec
            return max(total_time - overhead, 0.0)
        return total_time

    def _difficulty_prior_value(self, value_cfg: Any, difficulty_level: str, default: float = 0.0) -> float:
        """Read a scalar or difficulty-keyed shaping value."""
        if isinstance(value_cfg, dict):
            raw = value_cfg.get(difficulty_level, value_cfg.get("default", default))
        else:
            raw = value_cfg if value_cfg is not None else default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _difficulty_time_prior_shaping(
        self,
        problem: Problem,
        prev_time_sec: float,
        next_time_sec: float,
    ) -> float:
        """Optional reward shaping that encodes realistic time priors by difficulty.

        This is intentionally incremental: each allocation step receives a small
        bonus while moving toward the target time and a penalty once it exceeds it.
        """
        cfg = self.reward_cfg.get("allocation", {}).get("difficulty_time_prior", {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return 0.0

        difficulty_level = str(problem.difficulty_level)
        target_sec = self._difficulty_prior_value(cfg.get("target_sec"), difficulty_level, default=0.0)
        if target_sec <= 0.0:
            return 0.0

        unit = max(float(self.action_time_unit_sec), 1.0)
        reward = 0.0

        under_time = max(min(next_time_sec, target_sec) - prev_time_sec, 0.0)
        if under_time > 0.0:
            bonus_scale = self._difficulty_prior_value(
                cfg.get("under_target_bonus_scale"),
                difficulty_level,
                default=0.0,
            )
            reward += bonus_scale * (under_time / unit)

        over_before = max(prev_time_sec - target_sec, 0.0)
        over_after = max(next_time_sec - target_sec, 0.0)
        over_increment = max(over_after - over_before, 0.0)
        if over_increment > 0.0:
            penalty_scale = self._difficulty_prior_value(
                cfg.get("over_target_penalty_scale"),
                difficulty_level,
                default=0.0,
            )
            power = max(float(cfg.get("over_target_penalty_power", 1.0)), 0.0)
            over_ratio = max(over_after / max(target_sec, unit), 1e-9)
            reward += penalty_scale * (over_increment / unit) * (over_ratio ** max(power - 1.0, 0.0))

        return float(reward)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        # Select exam
        exam_index = options.get("exam_index")
        exam_slot = (
            int(self.rng.integers(0, len(self.exam_bank)))
            if exam_index is None
            else int(exam_index) % len(self.exam_bank)
        )
        selected = self.exam_bank[exam_slot]
        self.exam_data_path = str(selected["path"])
        original_problems = list(selected["problems"])

        self.problem_order = list(range(len(original_problems)))
        if self.shuffle_problem_order_on_reset and len(original_problems) > 1:
            self.problem_order = [int(i) for i in self.rng.permutation(len(original_problems))]
        self.problems = [original_problems[i] for i in self.problem_order]
        self.num_problems = len(self.problems)

        exam_total_time = float(
            self.exam_cfg.get("total_time_sec", selected["exam_data"].get("total_time_sec", 6000))
        )
        available_time = self._compute_available_time(exam_total_time, self.num_problems)

        # Safety: re-check max_steps with actual budget
        if self.max_steps_cfg is None:
            self.max_steps = (
                int(math.ceil(available_time / max(self.action_time_unit_sec, 1.0))) + self.num_problems
            )

        # Select student
        student_level = options.get("student_level", self.fixed_student_level)
        student_id = options.get("student_id", self.fixed_student_id)
        explicit_student = options.get("student_profile")

        if isinstance(explicit_student, StudentProfile):
            self.current_student = explicit_student
        elif isinstance(student_id, str):
            picked = next((s for s in self.student_profiles if s.student_id == student_id), None)
            if picked is None:
                raise ValueError(f"Student '{student_id}' not found in {self.student_data_path}")
            self.current_student = picked
        elif isinstance(student_level, str):
            self.current_student = create_level_profile(
                student_level, self.rng, preset_path=self.student_preset_path
            )
        elif self.student_profiles:
            self.current_student = sample_student_profile(self.student_profiles, self.rng)
        else:
            self.current_student = create_level_profile(
                "mid", self.rng, preset_path=self.student_preset_path
            )

        # Cache time-independent marginal gain params per problem
        _max_score = max((p.score for p in self.problems), default=1.0)
        self._mg_params_cache = [
            (
                *confidence_static_params(prob, self.current_student, self.dynamics_cfg),
                float(prob.score) / max(_max_score, 1.0),
            )
            for prob in self.problems
        ]

        progress = [ProblemProgress() for _ in range(self.num_problems)]
        for p, prob in zip(progress, self.problems):
            p.initialize_for_problem(prob)

        # current_problem_idx is tracked only for evaluate_trained_model() compat
        self.state = ExamState(
            remaining_time_sec=available_time,
            current_problem_idx=0,
            progress=progress,
            total_score=0.0,
            step_count=0,
            visit_order=[],
            same_problem_streak=0,
        )
        self.state.total_score = expected_total_score(self.state, self.problems)

        obs = self._get_obs()
        info = {
            "student_id": self.current_student.student_id,
            "exam_path": self.exam_data_path,
            "problem_order": list(self.problem_order),
            "problem_pids": [int(prob.pid) for prob in self.problems],
            "available_time_sec": available_time,
        }
        return obs, info

    def step(self, action):
        if self.state is None or self.current_student is None:
            raise RuntimeError("Call reset() before step().")

        if self._is_done():
            return self._get_obs(), 0.0, True, False, {"reason": "already_done"}

        problem_idx = int(np.asarray(action, dtype=np.int64).reshape(-1)[0]) % self.num_problems
        prev_score = float(self.state.total_score)

        progress = self.state.progress[problem_idx]
        problem = self.problems[problem_idx]

        spent = apply_time_cost(self.state, self.action_time_unit_sec)
        if spent > 0.0:
            prev_time_sec = float(progress.time_spent_sec)
            progress.status = ProblemStatus.IN_PROGRESS
            progress.time_spent_sec += spent
            next_time_sec = float(progress.time_spent_sec)

            scalar_conf = confidence_curve(
                problem=problem,
                student=self.current_student,
                time_spent=progress.time_spent_sec,
                dynamics_cfg=self.dynamics_cfg,
            )
            if problem.problem_type == "objective":
                scalar_conf = max(progress.effective_confidence(problem), scalar_conf)
            progress.sync_from_scalar(problem, scalar_conf)

        self.state.current_problem_idx = problem_idx
        self.state.total_score = expected_total_score(self.state, self.problems)
        self.state.step_count += 1

        reward = float(self.state.total_score) - prev_score
        if spent > 0.0:
            reward += self._difficulty_time_prior_shaping(
                problem=problem,
                prev_time_sec=prev_time_sec,
                next_time_sec=next_time_sec,
            )

        done = self._is_done()
        if done:
            reward += compute_terminal_reward(
                state=self.state,
                problems=self.problems,
                reward_cfg=self.reward_cfg,
                timed_out=self.state.remaining_time_sec <= 0,
                step_limited=self.state.step_count >= self.max_steps,
            )

        obs = self._get_obs()
        info = {
            "problem_idx": problem_idx,
            "remaining_time_sec": float(self.state.remaining_time_sec),
            "expected_score": float(self.state.total_score),
            "allocated_time_sec": [float(p.time_spent_sec) for p in self.state.progress],
            "problem_pids": [int(p.pid) for p in self.problems],
            "problem_order": list(self.problem_order),
        }
        return obs, reward, done, False, info

    def _is_done(self) -> bool:
        if self.state is None:
            return False
        return self.state.remaining_time_sec <= 0 or self.state.step_count >= self.max_steps

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        max_score = max(p.score for p in self.problems)
        # Denominator for time normalisation: use the initial budget (remaining + spent)
        total_spent = sum(float(p.time_spent_sec) for p in self.state.progress)
        time_denom = max(float(self.state.remaining_time_sec) + total_spent, 1.0)

        features: list[float] = [
            float(np.clip(self.state.remaining_time_sec / time_denom, 0.0, 1.0)),
        ]

        for i, progress in enumerate(self.state.progress):
            problem = self.problems[i]
            conf_slots = [float(np.clip(x, 0.0, 1.0)) for x in progress.confidence_slots(problem, width=5)]

            features.append(float(np.clip(progress.time_spent_sec / time_denom, 0.0, 1.0)))
            features.append(float(self.DIFFICULTY_LEVEL_MAP.get(problem.difficulty_level, 0.5)))
            features.append(float(np.clip(problem.score / max(max_score, 1), 0.0, 1.0)))
            features.append(float(self.PROBLEM_TYPE_MAP.get(problem.problem_type, 0.5)))
            features.append(float(np.clip(problem.error_rate, 0.0, 1.0)))
            features.extend(conf_slots)  # 5 slots

            # Marginal gain: expected score increase from one more action unit
            if self._mg_params_cache is not None:
                mg_floor, mg_static_logit, mg_alpha, mg_tau, mg_score_norm = self._mg_params_cache[i]
                t_future = float(progress.time_spent_sec) + self.action_time_unit_sec
                logit_f = mg_static_logit + mg_alpha * math.log(1.0 + t_future / max(mg_tau, 1.0))
                sig_f = (1.0 / (1.0 + math.exp(-logit_f)) if logit_f >= 0
                         else math.exp(logit_f) / (1.0 + math.exp(logit_f)))
                p_future = max(mg_floor, min(1.0, mg_floor + (1.0 - mg_floor) * sig_f))
                conf_now = float(progress.effective_confidence(problem))
                mg = float(np.clip(mg_score_norm * (p_future - conf_now), 0.0, 1.0))
            else:
                mg = 0.0
            features.append(mg)

        return np.asarray(features, dtype=np.float32)
