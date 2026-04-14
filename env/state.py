from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np

from .problem import Problem


class ProblemStatus(str, Enum):
    NOT_VISITED = "NOT_VISITED"
    IN_PROGRESS = "IN_PROGRESS"
    MOVED_ON = "MOVED_ON"

    @property
    def is_terminal(self) -> bool:
        return False


@dataclass
class ProblemProgress:
    status: ProblemStatus = ProblemStatus.NOT_VISITED
    time_spent_sec: float = 0.0
    answer_confidence: float = 0.0
    choice_confidences: list[float] = field(default_factory=list)

    @property
    def confidence_score(self) -> float:
        if self.choice_confidences:
            return max(float(x) for x in self.choice_confidences)
        return float(self.answer_confidence)

    @confidence_score.setter
    def confidence_score(self, value: float) -> None:
        self.answer_confidence = float(value)
        self.choice_confidences = []

    def _objective_distractor_weights(self, problem: Problem) -> list[float]:
        num_choices = max(problem.num_choices, 1)
        correct_idx = problem.correct_choice_index
        if correct_idx is None or correct_idx >= num_choices:
            correct_idx = 0

        if len(self.choice_confidences) == num_choices:
            weights = [max(float(v), 0.0) for v in self.choice_confidences]
        else:
            weights = []
            for i in range(num_choices):
                choice_key = str(i + 1)
                weights.append(float((problem.choice_rate or {}).get(choice_key, 1.0)))

        distractor_weights = [weights[i] for i in range(num_choices) if i != correct_idx]
        total = sum(distractor_weights)
        if total <= 0:
            return [1.0 for _ in distractor_weights]
        return [float(w / total) for w in distractor_weights]

    def update_objective_confidences(self, problem: Problem, correct_confidence: float) -> None:
        num_choices = max(problem.num_choices, 1)
        correct_idx = problem.correct_choice_index
        if correct_idx is None or correct_idx >= num_choices:
            correct_idx = 0

        correct = max(0.0, min(float(correct_confidence), 1.0))
        remaining = max(0.0, 1.0 - correct)
        distractor_weights = self._objective_distractor_weights(problem)

        confidences = [0.0 for _ in range(num_choices)]
        confidences[correct_idx] = correct
        distractor_indices = [i for i in range(num_choices) if i != correct_idx]
        for idx, weight in zip(distractor_indices, distractor_weights):
            confidences[idx] = remaining * float(weight)

        self.choice_confidences = confidences
        self.answer_confidence = correct

    def sync_from_scalar(self, problem: Problem, scalar_confidence: float) -> None:
        scalar = max(0.0, min(float(scalar_confidence), 1.0))
        self.answer_confidence = scalar
        if problem.problem_type != "objective":
            self.choice_confidences = []
            return
        self.update_objective_confidences(problem, scalar)

    def initialize_for_problem(self, problem: Problem) -> None:
        self.time_spent_sec = 0.0
        if problem.problem_type != "objective":
            self.answer_confidence = 0.02
            self.choice_confidences = []
            return

        num_choices = max(problem.num_choices, 1)
        correct_idx = problem.correct_choice_index
        if correct_idx is None or correct_idx >= num_choices:
            correct_idx = 0

        # Start with a mild prior for the correct choice rather than raw accuracy-like rates,
        # so the agent still benefits from additional solving time.
        base_correct_confidence = min(0.4, (1.0 / num_choices) + 0.1)
        self.choice_confidences = []
        self.update_objective_confidences(problem, base_correct_confidence)

    def predicted_choice_index(self) -> int | None:
        if not self.choice_confidences:
            return None
        return max(range(len(self.choice_confidences)), key=lambda i: float(self.choice_confidences[i]))

    def observable_confidence(self, problem: Problem) -> float:
        if problem.problem_type != "objective":
            return float(self.answer_confidence)
        if not self.choice_confidences:
            return float(self.answer_confidence)
        return max(float(x) for x in self.choice_confidences)

    def confidence_slots(self, problem: Problem, width: int = 5) -> list[float]:
        slots = [0.0] * max(int(width), 1)
        if problem.problem_type == "objective":
            for idx, confidence in enumerate(self.choice_confidences[: len(slots)]):
                slots[idx] = float(confidence)
            return slots
        slots[0] = float(self.answer_confidence)
        return slots

    def effective_confidence(self, problem: Problem) -> float:
        if problem.problem_type != "objective":
            return float(self.answer_confidence)
        correct_idx = problem.correct_choice_index
        if correct_idx is None or not self.choice_confidences:
            return float(self.confidence_score)
        if correct_idx >= len(self.choice_confidences):
            return 0.0
        return float(self.choice_confidences[correct_idx])

    def is_solved(
        self,
        problem: Problem,
        *,
        subjective_threshold: float = 0.5,
        objective_threshold: float = 0.5,
        objective_margin: float = 0.0,
    ) -> bool:
        if problem.problem_type != "objective":
            return self.effective_confidence(problem) >= float(subjective_threshold)
        correct_idx = problem.correct_choice_index
        predicted_idx = self.predicted_choice_index()
        if correct_idx is None or predicted_idx is None:
            return False
        if correct_idx >= len(self.choice_confidences):
            return False
        sorted_confidences = sorted((float(x) for x in self.choice_confidences), reverse=True)
        second_best = sorted_confidences[1] if len(sorted_confidences) > 1 else 0.0
        return (
            predicted_idx == correct_idx
            and self.effective_confidence(problem) >= float(objective_threshold)
            and (self.effective_confidence(problem) - second_best) >= float(objective_margin)
        )


def solved_criteria_from_config(config: dict[str, Any] | None) -> dict[str, float]:
    evaluation_cfg = (config or {}).get("evaluation", {})
    solved_cfg = evaluation_cfg.get("solved", {})
    return {
        "subjective_threshold": float(solved_cfg.get("subjective_conf_threshold", 0.5)),
        "objective_threshold": float(solved_cfg.get("objective_conf_threshold", 0.5)),
        "objective_margin": float(solved_cfg.get("objective_margin_threshold", 0.0)),
    }


@dataclass
class ExamState:
    remaining_time_sec: float
    current_problem_idx: int
    progress: list[ProblemProgress] = field(default_factory=list)
    total_score: float = 0.0
    step_count: int = 0
    visit_order: list[int] = field(default_factory=list)
    same_problem_streak: int = 0

    def is_all_terminal(self) -> bool:
        return all(p.status.is_terminal for p in self.progress)

    def is_all_visited(self) -> bool:
        return all(p.status != ProblemStatus.NOT_VISITED for p in self.progress)

    def visited_count(self) -> int:
        return sum(1 for p in self.progress if p.status != ProblemStatus.NOT_VISITED)

    def coverage_fraction(self) -> float:
        return self.visited_count() / max(len(self.progress), 1)

    def solved_count(self, problems: Iterable[Problem] | None = None, **criteria: float) -> int:
        if problems is None:
            return sum(1 for p in self.progress if p.confidence_score >= 0.5)
        problem_list = list(problems)
        return sum(1 for progress, problem in zip(self.progress, problem_list) if progress.is_solved(problem, **criteria))

    def subjective_problem_indices(self, problems: Iterable[Problem]) -> list[int]:
        problem_list = list(problems)
        return [idx for idx, problem in enumerate(problem_list) if problem.problem_type != "objective"]

    def objective_problem_indices(self, problems: Iterable[Problem]) -> list[int]:
        problem_list = list(problems)
        return [idx for idx, problem in enumerate(problem_list) if problem.problem_type == "objective"]

    def mean_subjective_confidence(self, problems: Iterable[Problem]) -> float:
        problem_list = list(problems)
        indices = self.subjective_problem_indices(problem_list)
        if not indices:
            return 0.0
        return float(np.mean([float(self.progress[idx].answer_confidence) for idx in indices]))

    def objective_dominance_rate(self, problems: Iterable[Problem]) -> float:
        problem_list = list(problems)
        indices = self.objective_problem_indices(problem_list)
        if not indices:
            return 0.0
        dominated = 0
        for idx in indices:
            progress = self.progress[idx]
            problem = problem_list[idx]
            predicted = progress.predicted_choice_index()
            correct_idx = problem.correct_choice_index
            if predicted is not None and correct_idx is not None and predicted == correct_idx:
                dominated += 1
        return float(dominated / len(indices))

    def subjective_solved_rate(self, problems: Iterable[Problem], **criteria: float) -> float:
        problem_list = list(problems)
        indices = self.subjective_problem_indices(problem_list)
        if not indices:
            return 0.0
        solved = 0
        for idx in indices:
            if self.progress[idx].is_solved(problem_list[idx], **criteria):
                solved += 1
        return float(solved / len(indices))

    def objective_solved_rate(self, problems: Iterable[Problem], **criteria: float) -> float:
        problem_list = list(problems)
        indices = self.objective_problem_indices(problem_list)
        if not indices:
            return 0.0
        solved = 0
        for idx in indices:
            if self.progress[idx].is_solved(problem_list[idx], **criteria):
                solved += 1
        return float(solved / len(indices))
