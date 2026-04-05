from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
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

        class _FallbackMultiDiscrete:
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec, dtype=np.int64)

        class _FallbackSpaces:
            Box = _FallbackBox
            MultiDiscrete = _FallbackMultiDiscrete

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()

from .dynamics import give_up_problem, solve_more, submit_answer
from .problem import Problem, load_exam_json
from .reward import compute_step_reward, compute_terminal_reward
from .state import ExamState, ProblemProgress, ProblemStatus
from .student import StudentProfile, create_level_profile, load_student_profiles, sample_student_profile


class ExamStrategyEnv(gym.Env):
    metadata = {"render_modes": []}

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
        self.training_cfg = dict(config.get("training", {}))
        self.data_cfg = dict(config.get("data", {}))
        self.student_cfg = dict(config.get("student", {}))

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def _resolve_data_path(path: str | None, default_rel_path: str) -> str:
            if path is None:
                return os.path.join(project_root, default_rel_path)
            if os.path.isabs(path):
                return path
            return os.path.join(project_root, path)

        cfg_exam_path = self.data_cfg.get("exam_path")
        cfg_student_path = self.data_cfg.get("student_path")
        cfg_student_preset_path = self.data_cfg.get("student_preset_path")

        self.exam_data_path = _resolve_data_path(
            exam_data_path if exam_data_path is not None else cfg_exam_path,
            os.path.join("data", "mock_exam.json"),
        )
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

        exam_data = load_exam_json(self.exam_data_path)
        self.problems: list[Problem] = [Problem.from_dict(x) for x in exam_data["problems"]]
        self.num_problems = len(self.problems)
        self.total_time_sec = float(self.exam_cfg.get("total_time_sec", exam_data.get("total_time_sec", 6000)))
        self.action_time_unit_sec = float(self.exam_cfg.get("action_time_unit_sec", 60))
        self.reveal_difficulty = bool(self.exam_cfg.get("reveal_difficulty", False))

        self.rng = np.random.default_rng(random_seed)
        self.student_profiles = load_student_profiles(self.student_data_path)
        self.current_student: StudentProfile | None = None
        self.state: ExamState | None = None

        # [target_problem_idx, action_type]
        # action_type: 0=solve_more, 1=submit, 2=give_up, 3=skip
        self.action_space = spaces.MultiDiscrete([self.num_problems, 4])

        obs_dim = 2 + (self.num_problems * 4)
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

        if isinstance(explicit_student, StudentProfile):
            self.current_student = explicit_student
        elif isinstance(student_id, str):
            picked = next((s for s in self.student_profiles if s.student_id == student_id), None)
            if picked is None:
                raise ValueError(f"Student id '{student_id}' not found in {self.student_data_path}")
            self.current_student = picked
        elif isinstance(student_level, str):
            self.current_student = create_level_profile(
                student_level,
                self.rng,
                preset_path=self.student_preset_path,
            )
        elif self.student_profiles:
            self.current_student = sample_student_profile(self.student_profiles, self.rng)
        else:
            self.current_student = create_level_profile(
                "mid",
                self.rng,
                preset_path=self.student_preset_path,
            )

        self.state = ExamState(
            remaining_time_sec=self.total_time_sec,
            current_problem_idx=0,
            progress=[ProblemProgress() for _ in range(self.num_problems)],
            total_score=0.0,
            step_count=0,
        )

        obs = self._get_obs()
        info = {"student_id": self.current_student.student_id}
        return obs, info

    def step(self, action):
        if self.state is None or self.current_student is None:
            raise RuntimeError("Call reset() before step().")

        if self._is_done():
            obs = self._get_obs()
            return obs, 0.0, True, False, {"reason": "already_done"}

        action_arr = np.asarray(action, dtype=np.int64).flatten()
        if action_arr.shape[0] != 2:
            raise ValueError("Action must have 2 components: [problem_idx, action_type].")

        problem_idx = int(np.clip(action_arr[0], 0, self.num_problems - 1))
        action_type = int(np.clip(action_arr[1], 0, 3))
        self.state.current_problem_idx = problem_idx
        problem = self.problems[problem_idx]

        prev_state = copy.deepcopy(self.state)
        action_name = "skip"
        was_correct: bool | None = None
        estimated_prob: float | None = None

        if action_type == 0:
            action_name = "solve_more"
            solve_more(
                state=self.state,
                problem_idx=problem_idx,
                delta_time_sec=self.action_time_unit_sec,
            )
        elif action_type == 1:
            action_name = "submit"
            was_correct, estimated_prob = submit_answer(
                state=self.state,
                problem_idx=problem_idx,
                problem=problem,
                student=self.current_student,
                rng=self.rng,
            )
        elif action_type == 2:
            action_name = "give_up"
            give_up_problem(self.state, problem_idx)
        else:
            action_name = "skip"

        self.state.step_count += 1
        terminated = self._is_done()
        truncated = False

        reward = compute_step_reward(
            prev_state=prev_state,
            next_state=self.state,
            problem_idx=problem_idx,
            problem=problem,
            action_name=action_name,
            was_correct=was_correct,
            reward_cfg=self.reward_cfg,
        )
        if terminated:
            reward += compute_terminal_reward(
                state=self.state,
                problems=self.problems,
                reward_cfg=self.reward_cfg,
            )

        obs = self._get_obs()
        info = {
            "action_name": action_name,
            "was_correct": was_correct,
            "estimated_prob": estimated_prob,
            "remaining_time_sec": self.state.remaining_time_sec,
            "total_score": self.state.total_score,
            "solved_count": self.state.solved_count(),
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State is not initialized. Call reset().")

        features: list[float] = []
        features.append(float(np.clip(self.state.remaining_time_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)))
        features.append(float(np.clip(self.state.current_problem_idx / max(self.num_problems - 1, 1), 0.0, 1.0)))

        status_to_num = {
            ProblemStatus.NOT_VISITED: 0.0,
            ProblemStatus.IN_PROGRESS: 1.0 / 4.0,
            ProblemStatus.SOLVED: 2.0 / 4.0,
            ProblemStatus.FAILED: 3.0 / 4.0,
            ProblemStatus.GIVEN_UP: 1.0,
        }

        max_score = max(p.score for p in self.problems)
        for i, progress in enumerate(self.state.progress):
            problem = self.problems[i]
            features.append(status_to_num[progress.status])
            features.append(float(np.clip(progress.time_spent_sec / max(self.total_time_sec, 1.0), 0.0, 1.0)))
            features.append(float(np.clip(problem.score / max(max_score, 1), 0.0, 1.0)))
            if self.reveal_difficulty:
                # Optional ablation mode; default is hidden to model partial observability.
                features.append(float(np.clip(problem.difficulty, 0.0, 1.0)))
            else:
                features.append(float(np.clip(progress.submit_count / 3.0, 0.0, 1.0)))

        return np.asarray(features, dtype=np.float32)

    def _is_done(self) -> bool:
        if self.state is None:
            return False
        if self.state.remaining_time_sec <= 0:
            return True
        return self.state.is_all_terminal()
