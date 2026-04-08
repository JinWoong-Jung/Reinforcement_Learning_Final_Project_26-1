from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    confidence_score: float = 0.0


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

    def solved_count(self) -> int:
        return sum(1 for p in self.progress if p.confidence_score >= 0.5)
