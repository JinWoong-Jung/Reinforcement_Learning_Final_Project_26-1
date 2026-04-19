from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StudentProfile:
    student_id: str
    theta: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudentProfile":
        theta = raw.get("theta")
        if theta is None:
            theta = legacy_theta_from_skills(raw)
        return cls(
            student_id=str(raw.get("student_id", "unknown")),
            theta=float(theta),
        )


def legacy_theta_from_skills(raw: dict[str, Any]) -> float:
    """Backward-compatible conversion for older skill-based student JSON."""
    required = ("skill_global", "skill_speed", "skill_accuracy")
    if not all(key in raw for key in required):
        raise KeyError("Student profile must include 'theta' or all legacy skill fields.")

    skill_global = float(raw["skill_global"])
    skill_speed = float(raw["skill_speed"])
    skill_accuracy = float(raw["skill_accuracy"])
    ability = (
        0.45 * skill_global
        + 0.30 * skill_accuracy
        + 0.25 * skill_speed
    )
    return float(3.0 * (ability - 0.5))


def load_student_profiles(path: str) -> list[StudentProfile]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_students = data.get("students", [])
    return [StudentProfile.from_dict(raw) for raw in raw_students]


def sample_student_profile(
    profiles: list[StudentProfile], rng: np.random.Generator
) -> StudentProfile:
    if not profiles:
        raise ValueError("profiles must not be empty.")
    idx = int(rng.integers(0, len(profiles)))
    return profiles[idx]


def load_student_level_presets(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "levels" not in data or not isinstance(data["levels"], dict):
        raise ValueError("Student preset JSON must include a 'levels' object.")
    return data


def create_level_profile(
    level: str,
    rng: np.random.Generator | None = None,
    preset_path: str | None = None,
) -> StudentProfile:
    level_key = level.lower()
    if rng is None:
        rng = np.random.default_rng()

    if preset_path is None:
        preset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "student_level_presets.json")
        )

    preset_data = load_student_level_presets(preset_path)
    levels = preset_data.get("levels", {})
    if level_key not in levels:
        raise ValueError("level must be one of: low, mid, high")

    noise = preset_data.get("noise", {})
    base = dict(levels[level_key])
    if "theta" in base:
        theta_std = float(noise.get("theta_std", noise.get("skill_std", 0.03)))
        theta = float(np.clip(float(base["theta"]) + rng.normal(0, theta_std), -8.0, 8.0))
    else:
        theta = legacy_theta_from_skills(base)

    return StudentProfile(
        student_id=f"{level_key}_sample",
        theta=theta,
    )
