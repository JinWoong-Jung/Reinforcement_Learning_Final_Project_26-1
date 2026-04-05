from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Problem:
    pid: int
    difficulty: float
    score: int
    avg_time: float
    error_rate: float
    problem_type: str
    topic: str
    actual_answer: Any | None = None
    choice_rate: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Problem":
        return cls(
            pid=int(raw["pid"]),
            difficulty=float(raw["difficulty"]),
            score=int(raw["score"]),
            avg_time=float(raw["avg_time"]),
            error_rate=float(raw["error_rate"]),
            problem_type=str(raw["problem_type"]),
            topic=str(raw["topic"]),
            actual_answer=raw.get("actual_answer"),
            choice_rate={str(k): float(v) for k, v in raw.get("choice_rate", {}).items()} or None,
        )


def load_exam_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "problems" not in data or not isinstance(data["problems"], list):
        raise ValueError("Exam JSON must include a 'problems' list.")
    return data


def load_problem_list(path: str) -> list[Problem]:
    data = load_exam_json(path)
    return [Problem.from_dict(raw) for raw in data["problems"]]
