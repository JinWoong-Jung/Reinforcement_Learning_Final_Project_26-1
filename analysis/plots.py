from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _ensure_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_score_distribution(results: dict[str, Any], save_path: str) -> str:
    _ensure_matplotlib()
    records = results.get("episode_records", [])
    scores = [float(r["total_score"]) for r in records]

    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=15, alpha=0.85, edgecolor="black")
    ax.set_title(f"Score Distribution: {results.get('policy_name', 'policy')}")
    ax.set_xlabel("Total Score")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def plot_time_usage_pattern(results: dict[str, Any], save_path: str, max_episodes: int = 12) -> str:
    _ensure_matplotlib()
    records = results.get("episode_records", [])[:max_episodes]

    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(9, 5))
    for rec in records:
        x = np.asarray(rec.get("used_time_timeline", []), dtype=float)
        y = np.asarray(rec.get("score_timeline", []), dtype=float)
        if x.size and y.size:
            ax.plot(x, y, alpha=0.5, linewidth=1.3)

    ax.set_title(f"Time Usage Pattern: {results.get('policy_name', 'policy')}")
    ax.set_xlabel("Used Time (sec)")
    ax.set_ylabel("Accumulated Score")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def plot_problem_avg_time(results: dict[str, Any], save_path: str) -> str:
    _ensure_matplotlib()
    times = np.asarray(results.get("problem_avg_time", []), dtype=float)
    problem_ids = np.arange(1, len(times) + 1)

    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(problem_ids, times, color="#4C78A8")
    ax.set_title(f"Average Time per Problem: {results.get('policy_name', 'policy')}")
    ax.set_xlabel("Problem ID")
    ax.set_ylabel("Avg Time Spent (sec)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def plot_student_level_strategy_gap(results: dict[str, Any], save_path: str) -> str:
    _ensure_matplotlib()
    breakdown: dict[str, dict[str, float]] = results.get("student_level_breakdown", {})
    levels = ["low", "mid", "high"]
    levels = [x for x in levels if x in breakdown] or list(breakdown.keys())

    score = [float(breakdown[l]["mean_score"]) for l in levels]
    easy = [float(breakdown[l]["mean_easy_recovery_rate"]) for l in levels]
    hard = [float(breakdown[l]["mean_hard_time_ratio"]) for l in levels]

    x = np.arange(len(levels))
    w = 0.25

    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, score, width=w, label="Mean Score", color="#4C78A8")
    ax.bar(x, easy, width=w, label="Easy Recovery Rate", color="#F58518")
    ax.bar(x + w, hard, width=w, label="Hard Time Ratio", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_title(f"Strategy Gap by Student Level: {results.get('policy_name', 'policy')}")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path
