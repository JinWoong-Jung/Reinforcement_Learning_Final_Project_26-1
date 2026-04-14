from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Problem:
    pid: int
    difficulty_level: str
    difficulty: float
    score: int
    error_rate: float
    problem_type: str
    actual_answer: Any | None = None
    choice_rate: dict[str, float] | None = None
    correct_rate: float | None = None

    @property
    def num_choices(self) -> int:
        if self.problem_type == "objective" and self.choice_rate:
            return max(len(self.choice_rate), 1)
        return 1

    @property
    def correct_choice_index(self) -> int | None:
        if self.problem_type != "objective":
            return None
        try:
            answer = int(self.actual_answer)
        except (TypeError, ValueError):
            return None
        return max(answer - 1, 0)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Problem":
        raw_correct_rate = raw.get("correct_rate")
        return cls(
            pid=int(raw["pid"]),
            difficulty_level=str(raw.get("difficulty_level", "unknown")),
            difficulty=float(raw["difficulty"]),
            score=int(raw["score"]),
            error_rate=float(raw["error_rate"]),
            problem_type=str(raw["problem_type"]),
            actual_answer=raw.get("actual_answer"),
            choice_rate={str(k): float(v) for k, v in raw.get("choice_rate", {}).items()} or None,
            correct_rate=float(raw_correct_rate) if raw_correct_rate is not None else None,
        )


def choice_entropy(problem: Problem) -> float:
    """Normalized Shannon entropy of choice_rate distribution.

    Returns a value in [0, 1] where 1 means maximum ambiguity (uniform
    distribution) and 0 means no ambiguity.  Returns 0.0 for subjective
    problems or when choice_rate is unavailable.
    """
    if problem.problem_type != "objective" or not problem.choice_rate:
        return 0.0
    values = [v for v in problem.choice_rate.values() if v > 0.0]
    if not values:
        return 0.0
    total = sum(values)
    probs = [v / total for v in values]
    raw_entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(max(len(values), 2))
    return float(raw_entropy / max_entropy)


def top2_gap(problem: Problem) -> float:
    """Gap between the top-2 choice rates (sorted descending).

    Returns a value in [0, 1].  A large gap means one option dominates
    (low ambiguity); a small gap means two options are close (higher
    ambiguity).  Returns 1.0 for subjective problems (no ambiguity from
    choice structure).
    """
    if problem.problem_type != "objective" or not problem.choice_rate:
        return 1.0
    sorted_rates = sorted(problem.choice_rate.values(), reverse=True)
    if len(sorted_rates) < 2:
        return 1.0
    return float(sorted_rates[0] - sorted_rates[1])


def distractor_mass(problem: Problem) -> float:
    """Fraction of responses that went to non-correct choices.

    Returns a value in [0, 1].  A high value means most students chose
    wrong answers (highly distracting), which indicates ambiguity.
    Returns 0.0 for subjective problems.
    """
    if problem.problem_type != "objective" or not problem.choice_rate:
        return 0.0
    correct_idx = problem.correct_choice_index
    if correct_idx is None:
        return 0.0
    total = sum(problem.choice_rate.values())
    if total <= 0.0:
        return 0.0
    correct_key = str(correct_idx + 1)
    correct_mass = float(problem.choice_rate.get(correct_key, 0.0))
    return float(max(1.0 - correct_mass / total, 0.0))


def load_exam_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "problems" not in data or not isinstance(data["problems"], list):
        raise ValueError("Exam JSON must include a 'problems' list.")
    return data


def load_problem_list(path: str) -> list[Problem]:
    data = load_exam_json(path)
    return [Problem.from_dict(raw) for raw in data["problems"]]
