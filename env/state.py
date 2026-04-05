from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProblemStatus(str, Enum):
    NOT_VISITED = "NOT_VISITED"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMITTED = "SUBMITTED"
    GIVEN_UP = "GIVEN_UP"

    @property
    def is_terminal(self) -> bool:
        return self in {ProblemStatus.SUBMITTED, ProblemStatus.GIVEN_UP}


@dataclass
class ProblemProgress:
    status: ProblemStatus = ProblemStatus.NOT_VISITED
    time_spent_sec: float = 0.0
    submit_count: int = 0
    judged_correct: bool | None = None
    confidence_score: float = 0.0


@dataclass
class ExamState:
    remaining_time_sec: float
    current_problem_idx: int
    progress: list[ProblemProgress] = field(default_factory=list)
    total_score: float = 0.0
    step_count: int = 0

    def is_all_terminal(self) -> bool:
        return all(p.status.is_terminal for p in self.progress)

    def solved_count(self) -> int:
        return sum(1 for p in self.progress if p.judged_correct is True)
