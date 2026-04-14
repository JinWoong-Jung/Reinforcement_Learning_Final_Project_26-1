from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    class _FallbackEnv:
        metadata = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return None

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackDiscrete:
        def __init__(self, n):
            self.n = int(n)

    class _FallbackMultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.asarray(nvec, dtype=np.int64)

    class _FallbackSpaces:
        Box = _FallbackBox
        Discrete = _FallbackDiscrete
        MultiDiscrete = _FallbackMultiDiscrete

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()

from .dynamics import apply_time_cost, expected_total_score, move_next, solve_more
from .problem import Problem, load_exam_json
from .reward import compute_step_reward, compute_terminal_reward
from .state import ExamState, ProblemProgress, ProblemStatus, solved_criteria_from_config
from .student import StudentProfile, create_level_profile, load_student_profiles, sample_student_profile


class ExamStrategyEnv(gym.Env):
    metadata = {"render_modes": []}
    DIFFICULTY_LEVEL_MAP = {
        "하": 0.0,
        "중하": 0.2,
        "중": 0.4,
        "중상": 0.6,
        "상": 0.8,
        "최상": 1.0,
        "unknown": 0.5,
    }
    PROBLEM_TYPE_MAP = {
        "objective": 0.0,
        "subjective": 1.0,
    }

    def __init__(
        self,
        config: dict[str, Any],
        exam_data_path: str | None = None,
        student_data_path: str | None = None,
        fixed_student_level: str | None = None,
        fixed_student_id: str | None = None,
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

        def _resolve_data_path(path: str | None, default_rel_path: str) -> str:
            if path is None:
                return os.path.join(project_root, default_rel_path)
            if os.path.isabs(path):
                return path
            return os.path.join(project_root, path)

        cfg_exam_path = self.data_cfg.get("exam_path")
        cfg_exam_paths = list(self.data_cfg.get("exam_paths", []))
        cfg_student_path = self.data_cfg.get("student_path")
        cfg_student_preset_path = self.data_cfg.get("student_preset_path")
        if exam_data_path is not None:
            self.exam_data_paths = [_resolve_data_path(exam_data_path, os.path.join("data", "mock_exam.json"))]
        elif cfg_exam_paths:
            self.exam_data_paths = [_resolve_data_path(path, os.path.join("data", "mock_exam.json")) for path in cfg_exam_paths]
        else:
            self.exam_data_paths = [_resolve_data_path(cfg_exam_path, os.path.join("data", "mock_exam.json"))]
        self.exam_data_path = self.exam_data_paths[0]
        self.student_data_path = _resolve_data_path(
            student_data_path if student_data_path is not None else cfg_student_path,
            os.path.join("data", "mock_students.json"),
        )
        self.student_preset_path = _resolve_data_path(
            cfg_student_preset_path,
            os.path.join("data", "student_level_presets.json"),
        )
        self.fixed_student_level = fixed_student_level or self.student_cfg.get("fixed_level")
        self.fixed_student_id = fixed_student_id or self.student_cfg.get("fixed_id")

        self.exam_bank: list[dict[str, Any]] = []
        base_num_problems: int | None = None
        for path in self.exam_data_paths:
            exam_data = load_exam_json(path)
            problems = [Problem.from_dict(x) for x in exam_data["problems"]]
            if base_num_problems is None:
                base_num_problems = len(problems)
            elif len(problems) != base_num_problems:
                raise ValueError("All exam files must have the same number of problems.")
            self.exam_bank.append({"path": path, "exam_data": exam_data, "problems": problems})
        if not self.exam_bank:
            raise ValueError("At least one exam file must be provided.")

        self.problems: list[Problem] = list(self.exam_bank[0]["problems"])
        self.num_problems = len(self.problems)
        self.total_time_sec = float(self.exam_cfg.get("total_time_sec", self.exam_bank[0]["exam_data"].get("total_time_sec", 6000)))
        self.action_time_unit_sec = float(self.exam_cfg.get("action_time_unit_sec", 30))
        self.switch_time_sec = float(self.exam_cfg.get("switch_time_sec", 10))
        self.randomize_start_problem = bool(self.exam_cfg.get("randomize_start_problem", True))
        self.difficulty_time_priors_sec = {
            str(k): float(v) for k, v in self.exam_cfg.get("difficulty_time_priors_sec", {}).items()
        }
        self.difficulty_target_confidences = {
            str(k): float(v) for k, v in self.exam_cfg.get("difficulty_target_confidences", {}).items()
        }
        self.problem_type_target_bonus = {
            str(k): float(v) for k, v in self.exam_cfg.get("problem_type_target_bonus", {}).items()
        }
        self.first_pass_cfg = dict(self.exam_cfg.get("first_pass", {}))
        self.revisit_policy_cfg = dict(self.exam_cfg.get("revisit_policy", {}))
        self.solve_more_constraints_cfg = dict(self.exam_cfg.get("solve_more_constraints", {}))
        min_step_time = max(min(self.action_time_unit_sec, self.switch_time_sec), 1.0)
        default_max_steps = int(np.ceil(self.total_time_sec / min_step_time)) + (self.num_problems * 2)
        self.max_steps = int(self.max_steps_cfg) if self.max_steps_cfg is not None else default_max_steps
        self.reveal_difficulty = bool(self.exam_cfg.get("reveal_difficulty", False))
        self.allow_agent_start = bool(self.exam_cfg.get("allow_agent_selected_start_problem", False))
        self.solved_criteria = solved_criteria_from_config(config)

        self.rng = np.random.default_rng(random_seed)
        self.student_profiles = load_student_profiles(self.student_data_path)
        self.current_student: StudentProfile | None = None
        self.state: ExamState | None = None
        self._recent_problem_entries: list[int] = []
        self._current_session_had_work = False
        self._current_session_entry_was_revisit = False

        # [action_type, target_problem_idx]
        # action_type: 0=solve_more on current problem, 1=move to another problem
        # target_problem_idx: absolute 0-based index of the destination problem.
        # If action_type==1 and target_problem_idx==current, redirects to (current+1)%N.
        self.action_space = spaces.MultiDiscrete([2, self.num_problems])

        # Per-problem features (11): status, time_spent, difficulty_level, score,
        # problem_type, error_rate, and five confidence slots.
        # Confidence slots are type-aware but fixed-length:
        # - subjective: [answer_confidence, 0, 0, 0, 0]
        # - objective: [c1, c2, c3, c4, c5]
        # NOTE: pid and true difficulty remain hidden from the agent.
        obs_dim = 2 + (self.num_problems * 11)
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        student_level = options.get("student_level", self.fixed_student_level)
        student_id = options.get("student_id", self.fixed_student_id)
        explicit_student = options.get("student_profile")
        exam_index = options.get("exam_index")

        if exam_index is None:
            exam_slot = int(self.rng.integers(0, len(self.exam_bank)))
        else:
            exam_slot = int(exam_index) % len(self.exam_bank)
        selected_exam = self.exam_bank[exam_slot]
        self.exam_data_path = str(selected_exam["path"])
        self.problems = list(selected_exam["problems"])
        self.num_problems = len(self.problems)
        self.total_time_sec = float(self.exam_cfg.get("total_time_sec", selected_exam["exam_data"].get("total_time_sec", 6000)))
        min_step_time = max(min(self.action_time_unit_sec, self.switch_time_sec), 1.0)
        default_max_steps = int(np.ceil(self.total_time_sec / min_step_time)) + (self.num_problems * 2)
        self.max_steps = int(self.max_steps_cfg) if self.max_steps_cfg is not None else default_max_steps

        if isinstance(explicit_student, StudentProfile):
            self.current_student = explicit_student
        elif isinstance(student_id, str):
            picked = next((s for s in self.student_profiles if s.student_id == student_id), None)
            if picked is None:
                raise ValueError(f"Student id '{student_id}' not found in {self.student_data_path}")
            self.current_student = picked
        elif isinstance(student_level, str):
            self.current_student = create_level_profile(student_level, self.rng, preset_path=self.student_preset_path)
        elif self.student_profiles:
            self.current_student = sample_student_profile(self.student_profiles, self.rng)
        else:
            self.current_student = create_level_profile("mid", self.rng, preset_path=self.student_preset_path)

        progress = [ProblemProgress() for _ in range(self.num_problems)]
        for p, problem in zip(progress, self.problems):
            p.initialize_for_problem(problem)

        if self.allow_agent_start:
            # New mode: agent selects the start problem as its first action.
            # current_problem_idx = -1 is a sentinel meaning "not yet started".
            # All problems remain NOT_VISITED until the first step.
            start_idx = -1
            visit_order: list[int] = []
        else:
            # Old mode (default): start problem is fixed or randomised here.
            start_idx = 0
            if self.randomize_start_problem and self.num_problems > 0:
                start_idx = int(self.rng.integers(0, self.num_problems))
            progress[start_idx].status = ProblemStatus.IN_PROGRESS
            visit_order = [start_idx]

        self.state = ExamState(
            remaining_time_sec=self.total_time_sec,
            current_problem_idx=start_idx,
            progress=progress,
            total_score=0.0,
            step_count=0,
            visit_order=visit_order,
            same_problem_streak=0,
        )
        self.state.total_score = expected_total_score(self.state, self.problems)
        self._recent_problem_entries = [] if start_idx == -1 else [start_idx]
        self._current_session_had_work = False
        self._current_session_entry_was_revisit = False
        obs = self._get_obs()
        info = {"student_id": self.current_student.student_id, "exam_path": self.exam_data_path, "start_problem_idx": start_idx}
        return obs, info

    def step(self, action):
        if self.state is None or self.current_student is None:
            raise RuntimeError("Call reset() before step().")

        if self._is_done():
            return self._get_obs(), 0.0, True, False, {"reason": "already_done"}

        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.size != 2:
            raise ValueError("Action must be [action_type, next_target_choice].")
        action_id = int(action_arr[0])
        next_target_choice = int(action_arr[1])
        if action_id not in {0, 1}:
            raise ValueError("action_type must be 0 (solve_more) or 1 (next).")
        if not (0 <= next_target_choice < self.num_problems):
            raise ValueError(f"next_target_choice must be in [0, {self.num_problems - 1}]")

        # ── Not-started mode (allow_agent_selected_start_problem) ─────────────
        # The very first action selects which problem to start on.
        # next_target_choice picks the start problem; action_type is ignored.
        # No switch-time cost is charged for this initial selection.
        if self.state.current_problem_idx == -1:
            target_idx = int(np.clip(next_target_choice, 0, self.num_problems - 1))
            self.state.current_problem_idx = target_idx
            self.state.progress[target_idx].status = ProblemStatus.IN_PROGRESS
            self.state.visit_order.append(target_idx)
            self._recent_problem_entries = [target_idx]
            self._current_session_had_work = False
            self._current_session_entry_was_revisit = False
            self.state.total_score = expected_total_score(self.state, self.problems)
            self.state.step_count += 1
            info = {
                "action_name": "select_start",
                "target_problem_idx": target_idx,
                "remaining_time_sec": self.state.remaining_time_sec,
                "current_problem_idx": target_idx,
                "expected_score": self.state.total_score,
                "same_problem_streak": 0,
                "exam_path": self.exam_data_path,
                "visit_order": [target_idx + 1],
                "forced_switch_reason": None,
                "no_work_revisit": False,
            }
            return self._get_obs(), 0.0, self._is_done(), False, info
        # ────────────────────────────────────────────────────────────────────

        prev_state = copy.deepcopy(self.state)
        current_idx = self.state.current_problem_idx
        action_name = "solve_more" if action_id == 0 else "next"
        target_idx = current_idx
        forced_switch_reason: str | None = None
        no_work_revisit = False

        if action_id == 0:
            forced_switch_reason = self._solve_more_block_reason(current_idx)
        if action_id == 0 and forced_switch_reason is None:
            solve_more(
                state=self.state,
                problem_idx=current_idx,
                delta_time_sec=self.action_time_unit_sec,
                problem=self.problems[current_idx],
                student=self.current_student,
                total_time_sec=self.total_time_sec,
                dynamics_cfg=self.dynamics_cfg,
            )
            self.state.same_problem_streak += 1
            self._current_session_had_work = True
        else:
            if action_id == 0:
                action_name = "next"
            no_work_revisit = self._current_session_entry_was_revisit and not self._current_session_had_work
            apply_time_cost(self.state, self.switch_time_sec)
            target_idx = (
                self._forced_switch_target(current_idx)
                if forced_switch_reason is not None
                else self._decode_next_target(current_idx, next_target_choice)
            )
            target_progress = self.state.progress[target_idx]
            target_has_prior_work = float(target_progress.time_spent_sec) > 0.0
            move_next(self.state, current_idx, target_idx)
            self._record_problem_entry(target_idx)
            self.state.same_problem_streak = 0
            self._current_session_had_work = False
            self._current_session_entry_was_revisit = bool(target_has_prior_work)

        self.state.total_score = expected_total_score(self.state, self.problems)
        self.state.step_count += 1
        terminated = self._is_done()
        truncated = False

        reward = compute_step_reward(
            prev_state=prev_state,
            next_state=self.state,
            problems=self.problems,
            action_name=action_name,
            reward_cfg=self.reward_cfg,
        )
        if terminated:
            timed_out = self.state.remaining_time_sec <= 0
            step_limited = not timed_out and self.state.step_count >= self.max_steps
            reward += compute_terminal_reward(
                self.state,
                self.problems,
                self.reward_cfg,
                timed_out=timed_out,
                step_limited=step_limited,
            )
        if action_name == "next" and no_work_revisit:
            reward += float(self.reward_cfg.get("next", {}).get("no_work_revisit_penalty", 0.0))

        info = {
            "action_name": action_name,
            "target_problem_idx": target_idx,
            "remaining_time_sec": self.state.remaining_time_sec,
            "current_problem_idx": self.state.current_problem_idx,
            "expected_score": self.state.total_score,
            "same_problem_streak": self.state.same_problem_streak,
            "exam_path": self.exam_data_path,
            "visit_order": [idx + 1 for idx in self.state.visit_order],
            "forced_switch_reason": forced_switch_reason,
            "no_work_revisit": bool(action_name == "next" and no_work_revisit),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _other_problem_indices(self, current_idx: int) -> list[int]:
        return [idx for idx in range(self.num_problems) if idx != current_idx]

    def _decode_next_target(self, current_idx: int, target_idx: int) -> int:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")
        target = int(np.clip(target_idx, 0, self.num_problems - 1))
        sequential_unvisited = self._next_unvisited_from(current_idx)
        if target == current_idx:
            if sequential_unvisited is not None:
                target = sequential_unvisited
            else:
                priority_revisit = self._priority_revisit_target(current_idx)
                target = priority_revisit if priority_revisit is not None else (current_idx + 1) % self.num_problems
        if (
            bool(self.first_pass_cfg.get("enabled", False))
            and bool(self.first_pass_cfg.get("enforce_unvisited_before_revisit", False))
            and sequential_unvisited is not None
            and self.state.progress[target].status != ProblemStatus.NOT_VISITED
        ):
            return sequential_unvisited
        return target

    def encode_solve_more_action(self) -> np.ndarray:
        return np.array([0, 0], dtype=np.int64)

    def encode_select_start_action(self, target_problem_idx: int) -> np.ndarray:
        """Return the action that selects the initial problem in allow_agent_start mode.

        Only meaningful when current_problem_idx == -1 (not-started state).
        action_type is ignored in that state; next_target_choice picks the problem.
        """
        if not (0 <= target_problem_idx < self.num_problems):
            raise ValueError(f"target_problem_idx must be in [0, {self.num_problems - 1}].")
        return np.array([1, int(target_problem_idx)], dtype=np.int64)

    def encode_next_action(self, target_problem_idx: int) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")
        if not (0 <= target_problem_idx < self.num_problems):
            raise ValueError(f"target_problem_idx must be in [0, {self.num_problems - 1}].")
        if target_problem_idx == self.state.current_problem_idx:
            raise ValueError("target_problem_idx must be different from the current problem.")
        return np.array([1, int(target_problem_idx)], dtype=np.int64)

    def _solve_more_block_reason(self, current_idx: int) -> str | None:
        if self.state is None:
            return None
        cfg = self.solve_more_constraints_cfg
        if not bool(cfg.get("enabled", False)):
            return None

        progress = self.state.progress[current_idx]
        problem = self.problems[current_idx]

        streak_threshold = int(cfg.get("streak_threshold", -1))
        if streak_threshold >= 0 and self.state.same_problem_streak >= streak_threshold:
            return "streak_limit"

        difficulty_budget = self._difficulty_time_budget_sec(problem)
        if (
            difficulty_budget is not None
            and self._has_unvisited_other_than(current_idx)
            and progress.time_spent_sec >= difficulty_budget
        ):
            return "difficulty_time_budget_reached"

        if problem.problem_type == "objective":
            threshold = float(cfg.get("objective_conf_threshold", 1.1))
            if progress.effective_confidence(problem) >= threshold:
                return "objective_conf_saturated"
            return None

        threshold = float(cfg.get("subjective_conf_threshold", 1.1))
        if progress.answer_confidence >= threshold:
            return "subjective_conf_saturated"
        return None

    def _forced_switch_target(self, current_idx: int) -> int:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")
        sequential_unvisited = self._next_unvisited_from(current_idx)
        if sequential_unvisited is not None:
            return sequential_unvisited
        priority_revisit = self._priority_revisit_target(current_idx)
        if priority_revisit is not None:
            return priority_revisit
        return int((current_idx + 1) % self.num_problems)

    def _difficulty_time_budget_sec(self, problem: Problem) -> float | None:
        if not self.difficulty_time_priors_sec:
            return None
        budget = self.difficulty_time_priors_sec.get(problem.difficulty_level)
        if budget is None:
            return None
        return float(budget)

    def _has_unvisited_other_than(self, current_idx: int) -> bool:
        if self.state is None:
            return False
        return any(
            idx != current_idx and progress.status == ProblemStatus.NOT_VISITED
            for idx, progress in enumerate(self.state.progress)
        )

    def _next_unvisited_from(self, current_idx: int) -> int | None:
        if self.state is None:
            return None
        if not bool(self.first_pass_cfg.get("enabled", False)):
            candidates = [
                idx for idx, progress in enumerate(self.state.progress)
                if idx != current_idx and progress.status == ProblemStatus.NOT_VISITED
            ]
            return int(candidates[0]) if candidates else None

        indices = list(range(current_idx + 1, self.num_problems)) + list(range(0, current_idx))
        for idx in indices:
            if self.state.progress[idx].status == ProblemStatus.NOT_VISITED:
                return int(idx)
        return None

    def _priority_revisit_target(self, current_idx: int) -> int | None:
        if self.state is None:
            return None
        cooldown_entries = int(self.revisit_policy_cfg.get("cooldown_entries", 2))
        prefer_worked = bool(self.revisit_policy_cfg.get("prefer_worked_problems", True))
        recent_blocked = set(self._recent_problem_entries[-cooldown_entries:]) if cooldown_entries > 0 else set()
        candidates = []
        cooled_candidates = []
        for idx, progress in enumerate(self.state.progress):
            if idx == current_idx or progress.status == ProblemStatus.NOT_VISITED:
                continue
            problem = self.problems[idx]
            is_solved = progress.is_solved(problem, **self.solved_criteria)
            difficulty_rank = float(self.DIFFICULTY_LEVEL_MAP.get(problem.difficulty_level, 0.5))
            effective_conf = float(progress.effective_confidence(problem))
            has_work = float(progress.time_spent_sec > 0.0)
            candidate = (
                (
                    1 if not is_solved else 0,
                    1 if (prefer_worked and has_work > 0.0) else 0,
                    difficulty_rank,
                    float(problem.score),
                    -effective_conf,
                    float(progress.time_spent_sec),
                    -abs(idx - current_idx),
                    -idx,
                    idx,
                )
            )
            candidates.append(candidate)
            if idx not in recent_blocked:
                cooled_candidates.append(candidate)
        pool = cooled_candidates or candidates
        if not pool:
            return None
        _, _, _, _, _, _, _, _, best_idx = max(pool)
        return int(best_idx)

    def _record_problem_entry(self, problem_idx: int) -> None:
        self._recent_problem_entries.append(int(problem_idx))
        max_history = max(int(self.revisit_policy_cfg.get("cooldown_entries", 2)) + 3, 8)
        if len(self._recent_problem_entries) > max_history:
            self._recent_problem_entries = self._recent_problem_entries[-max_history:]

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")

        status_to_num = {
            ProblemStatus.NOT_VISITED: 0.0,
            ProblemStatus.IN_PROGRESS: 0.5,
            ProblemStatus.MOVED_ON: 1.0,
        }

        max_score = max(p.score for p in self.problems)
        # When current_problem_idx == -1 (not-started mode), encode as 0.0.
        # The agent can still infer the "not started" state from all statuses being NOT_VISITED.
        safe_current_idx = max(self.state.current_problem_idx, 0)
        features: list[float] = [
            float(np.clip(self.state.remaining_time_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)),
            float(np.clip(safe_current_idx / max(self.num_problems - 1, 1), 0.0, 1.0)),
        ]

        for i, progress in enumerate(self.state.progress):
            problem = self.problems[i]
            confidence_slots = [float(np.clip(x, 0.0, 1.0)) for x in progress.confidence_slots(problem, width=5)]

            features.append(status_to_num[progress.status])
            features.append(float(np.clip(progress.time_spent_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)))
            features.append(float(self.DIFFICULTY_LEVEL_MAP.get(problem.difficulty_level, 0.5)))
            features.append(float(np.clip(problem.score / max(max_score, 1), 0.0, 1.0)))
            features.append(float(self.PROBLEM_TYPE_MAP.get(problem.problem_type, 0.5)))
            features.append(float(np.clip(problem.error_rate, 0.0, 1.0)))
            features.extend(confidence_slots)

        return np.asarray(features, dtype=np.float32)

    def _is_done(self) -> bool:
        if self.state is None:
            return False
        if self.state.remaining_time_sec <= 0:
            return True
        if self.state.step_count >= self.max_steps:
            return True
        return False
