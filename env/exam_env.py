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
from .state import ExamState, ProblemProgress, ProblemStatus
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
        min_step_time = max(min(self.action_time_unit_sec, self.switch_time_sec), 1.0)
        default_max_steps = int(np.ceil(self.total_time_sec / min_step_time)) + (self.num_problems * 2)
        self.max_steps = int(self.max_steps_cfg) if self.max_steps_cfg is not None else default_max_steps
        self.reveal_difficulty = bool(self.exam_cfg.get("reveal_difficulty", False))

        self.rng = np.random.default_rng(random_seed)
        self.student_profiles = load_student_profiles(self.student_data_path)
        self.current_student: StudentProfile | None = None
        self.state: ExamState | None = None

        # [action_type, target_problem_idx]
        # action_type: 0=solve_more on current problem, 1=move to another problem
        # target_problem_idx: absolute 0-based index of the destination problem.
        # If action_type==1 and target_problem_idx==current, redirects to (current+1)%N.
        self.action_space = spaces.MultiDiscrete([2, self.num_problems])

        # Per-problem features (7): status, time_spent, difficulty_level, score,
        # problem_type, pid, error_rate.
        # NOTE: confidence_score is intentionally excluded — it is an internal
        # dynamics estimate derived from hidden student attributes and must not
        # be observable by the agent.
        obs_dim = 2 + (self.num_problems * 7)
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

        start_idx = 0
        if self.randomize_start_problem and self.num_problems > 0:
            start_idx = int(self.rng.integers(0, self.num_problems))
        progress = [ProblemProgress() for _ in range(self.num_problems)]
        progress[start_idx].status = ProblemStatus.IN_PROGRESS
        self.state = ExamState(
            remaining_time_sec=self.total_time_sec,
            current_problem_idx=start_idx,
            progress=progress,
            total_score=0.0,
            step_count=0,
            visit_order=[start_idx],
            same_problem_streak=0,
        )
        self.state.total_score = expected_total_score(self.state, self.problems)
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

        prev_state = copy.deepcopy(self.state)
        current_idx = self.state.current_problem_idx
        action_name = "solve_more" if action_id == 0 else "next"
        target_idx = current_idx

        if action_id == 0:
            solve_more(
                state=self.state,
                problem_idx=current_idx,
                delta_time_sec=self.action_time_unit_sec,
                problem=self.problems[current_idx],
                student=self.current_student,
                total_time_sec=self.total_time_sec,
            )
            self.state.same_problem_streak += 1
        else:
            apply_time_cost(self.state, self.switch_time_sec)
            target_idx = self._decode_next_target(current_idx, next_target_choice)
            move_next(self.state, current_idx, target_idx)
            self.state.same_problem_streak = 0

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

        info = {
            "action_name": action_name,
            "target_problem_idx": target_idx,
            "remaining_time_sec": self.state.remaining_time_sec,
            "current_problem_idx": self.state.current_problem_idx,
            "expected_score": self.state.total_score,
            "same_problem_streak": self.state.same_problem_streak,
            "exam_path": self.exam_data_path,
            "visit_order": [idx + 1 for idx in self.state.visit_order],
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _other_problem_indices(self, current_idx: int) -> list[int]:
        return [idx for idx in range(self.num_problems) if idx != current_idx]

    def _decode_next_target(self, current_idx: int, target_idx: int) -> int:
        target = int(np.clip(target_idx, 0, self.num_problems - 1))
        if target == current_idx:
            # Agent selected the current problem; redirect to the next one.
            target = (current_idx + 1) % self.num_problems
        return target

    def encode_solve_more_action(self) -> np.ndarray:
        return np.array([0, 0], dtype=np.int64)

    def encode_next_action(self, target_problem_idx: int) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")
        if not (0 <= target_problem_idx < self.num_problems):
            raise ValueError(f"target_problem_idx must be in [0, {self.num_problems - 1}].")
        if target_problem_idx == self.state.current_problem_idx:
            raise ValueError("target_problem_idx must be different from the current problem.")
        return np.array([1, int(target_problem_idx)], dtype=np.int64)

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")

        status_to_num = {
            ProblemStatus.NOT_VISITED: 0.0,
            ProblemStatus.IN_PROGRESS: 0.5,
            ProblemStatus.MOVED_ON: 1.0,
        }

        max_score = max(p.score for p in self.problems)
        max_pid = max(p.pid for p in self.problems)
        features: list[float] = [
            float(np.clip(self.state.remaining_time_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)),
            float(np.clip(self.state.current_problem_idx / max(self.num_problems - 1, 1), 0.0, 1.0)),
        ]

        for i, progress in enumerate(self.state.progress):
            problem = self.problems[i]
            features.append(status_to_num[progress.status])
            features.append(float(np.clip(progress.time_spent_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)))
            features.append(float(self.DIFFICULTY_LEVEL_MAP.get(problem.difficulty_level, 0.5)))
            features.append(float(np.clip(problem.score / max(max_score, 1), 0.0, 1.0)))
            features.append(float(self.PROBLEM_TYPE_MAP.get(problem.problem_type, 0.5)))
            features.append(float(np.clip(problem.pid / max(max_pid, 1), 0.0, 1.0)))
            features.append(float(np.clip(problem.error_rate, 0.0, 1.0)))

        return np.asarray(features, dtype=np.float32)

    def _is_done(self) -> bool:
        if self.state is None:
            return False
        if self.state.remaining_time_sec <= 0:
            return True
        if self.state.step_count >= self.max_steps:
            return True
        return False
