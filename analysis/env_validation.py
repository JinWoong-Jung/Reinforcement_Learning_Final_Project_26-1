from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.dynamics import confidence_curve, guessing_prob
from env.exam_env import ExamStrategyEnv
from env.problem import Problem, choice_entropy, top2_gap
from env.reward import compute_step_reward, expected_utility
from env.state import ExamState, ProblemProgress, ProblemStatus
from env.student import create_level_profile
from utils.io import load_config, save_json


@dataclass(frozen=True)
class ProblemCase:
    tag: str
    exam_path: str
    problem: Problem
    anchor: float
    entropy: float
    expected_max_score: float

    @property
    def label(self) -> str:
        return f"{self.tag}: {os.path.basename(self.exam_path)} / pid={self.problem.pid}"

    def describe(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "exam_path": self.exam_path,
            "pid": int(self.problem.pid),
            "problem_type": self.problem.problem_type,
            "difficulty_level": self.problem.difficulty_level,
            "difficulty": float(self.problem.difficulty),
            "correct_rate": None if self.problem.correct_rate is None else float(self.problem.correct_rate),
            "error_rate": float(self.problem.error_rate),
            "score": int(self.problem.score),
            "anchor": float(self.anchor),
            "choice_entropy": float(self.entropy),
            "top2_gap": float(top2_gap(self.problem)),
        }


def _ensure_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for env validation plots. Install with: pip install matplotlib")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg.setdefault("data", {})
    if args.exam_data:
        cfg["data"]["exam_path"] = args.exam_data
        cfg["data"].pop("exam_paths", None)
    if args.student_data:
        cfg["data"]["student_path"] = args.student_data
    cfg.setdefault("student", {})
    if args.student_id:
        cfg["student"]["fixed_id"] = args.student_id
    if args.student_level:
        cfg["student"]["fixed_level"] = args.student_level
    student_path = cfg["data"].get("student_path")
    if not student_path:
        fallback_student_path = os.path.join("data", "someone.json")
        if os.path.exists(os.path.join(PROJECT_ROOT, fallback_student_path)):
            cfg["data"]["student_path"] = fallback_student_path
    elif not os.path.isabs(student_path):
        candidate = os.path.join(PROJECT_ROOT, student_path)
        if not os.path.exists(candidate):
            fallback_student_path = os.path.join("data", "someone.json")
            if os.path.exists(os.path.join(PROJECT_ROOT, fallback_student_path)):
                cfg["data"]["student_path"] = fallback_student_path
    return cfg


def _difficulty_anchor(problem: Problem, dynamics_cfg: dict[str, Any]) -> float:
    anchor_source = str(dynamics_cfg.get("anchor_source", "difficulty"))
    if anchor_source == "correct_rate" and problem.correct_rate is not None:
        return float(max(0.0, min(1.0, 1.0 - problem.correct_rate)))
    return float(max(0.0, min(1.0, problem.difficulty)))


def _load_env_and_student(config: dict[str, Any], args: argparse.Namespace) -> tuple[ExamStrategyEnv, Any]:
    env = ExamStrategyEnv(config=config, random_seed=args.seed)
    reset_options: dict[str, Any] = {}
    if args.exam_index is not None:
        reset_options["exam_index"] = int(args.exam_index)
    if args.student_id:
        reset_options["student_id"] = args.student_id
    elif args.student_level:
        reset_options["student_level"] = args.student_level
    elif config.get("student", {}).get("fixed_id"):
        reset_options["student_id"] = config["student"]["fixed_id"]
    elif config.get("student", {}).get("fixed_level"):
        reset_options["student_level"] = config["student"]["fixed_level"]
    else:
        reset_options["student_level"] = "mid"
    env.reset(seed=args.seed, options=reset_options)
    assert env.current_student is not None
    return env, env.current_student


def _all_problem_cases(env: ExamStrategyEnv, dynamics_cfg: dict[str, Any]) -> list[ProblemCase]:
    cases: list[ProblemCase] = []
    for bank in env.exam_bank:
        exam_path = str(bank["path"])
        for problem in bank["problems"]:
            anchor = _difficulty_anchor(problem, dynamics_cfg)
            entropy = float(choice_entropy(problem))
            cases.append(
                ProblemCase(
                    tag="",
                    exam_path=exam_path,
                    problem=problem,
                    anchor=anchor,
                    entropy=entropy,
                    expected_max_score=float(problem.score),
                )
            )
    return cases


def _choose_extreme(cases: list[ProblemCase], *, problem_type: str, easiest: bool) -> ProblemCase:
    pool = [c for c in cases if c.problem.problem_type == problem_type]
    if not pool:
        raise ValueError(f"No {problem_type} problems found in loaded exams.")
    ranked = sorted(
        pool,
        key=lambda c: (
            c.anchor,
            c.entropy,
            -c.problem.score,
            c.problem.pid,
        ),
        reverse=not easiest,
    )
    return ranked[0]


def _choose_ambiguity_pair(cases: list[ProblemCase]) -> tuple[ProblemCase, ProblemCase]:
    pool = [c for c in cases if c.problem.problem_type == "objective"]
    if len(pool) < 2:
        raise ValueError("Need at least two objective problems to build ambiguity pair.")
    best_pair: tuple[ProblemCase, ProblemCase] | None = None
    best_score = -math.inf
    for low in pool:
        for high in pool:
            if high.entropy <= low.entropy:
                continue
            entropy_gap = high.entropy - low.entropy
            anchor_penalty = abs(high.anchor - low.anchor)
            score_penalty = abs(float(high.problem.score) - float(low.problem.score)) / 10.0
            pair_score = entropy_gap - (1.5 * anchor_penalty) - score_penalty
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (low, high)
    if best_pair is None:
        ordered = sorted(pool, key=lambda c: c.entropy)
        return ordered[0], ordered[-1]
    return best_pair


def _choose_matched_cross_type_pair(cases: list[ProblemCase]) -> tuple[ProblemCase, ProblemCase]:
    objective_cases = [c for c in cases if c.problem.problem_type == "objective"]
    subjective_cases = [c for c in cases if c.problem.problem_type == "subjective"]
    if not objective_cases or not subjective_cases:
        raise ValueError("Need both objective and subjective problems to compare type difference.")
    best_pair: tuple[ProblemCase, ProblemCase] | None = None
    best_distance = math.inf
    for objective in objective_cases:
        for subjective in subjective_cases:
            distance = (
                abs(objective.anchor - subjective.anchor)
                + 0.10 * abs(float(objective.problem.score) - float(subjective.problem.score))
            )
            if distance < best_distance:
                best_distance = distance
                best_pair = (objective, subjective)
    assert best_pair is not None
    return best_pair


def _retag(case: ProblemCase, tag: str) -> ProblemCase:
    return ProblemCase(
        tag=tag,
        exam_path=case.exam_path,
        problem=case.problem,
        anchor=case.anchor,
        entropy=case.entropy,
        expected_max_score=case.expected_max_score,
    )


def _selected_cases(cases: list[ProblemCase]) -> dict[str, ProblemCase]:
    obj_easy = _retag(_choose_extreme(cases, problem_type="objective", easiest=True), "objective_easy")
    obj_hard = _retag(_choose_extreme(cases, problem_type="objective", easiest=False), "objective_hard")
    subj_easy = _retag(_choose_extreme(cases, problem_type="subjective", easiest=True), "subjective_easy")
    subj_hard = _retag(_choose_extreme(cases, problem_type="subjective", easiest=False), "subjective_hard")
    low_amb, high_amb = _choose_ambiguity_pair(cases)
    matched_obj, matched_subj = _choose_matched_cross_type_pair(cases)
    return {
        "objective_easy": obj_easy,
        "objective_hard": obj_hard,
        "subjective_easy": subj_easy,
        "subjective_hard": subj_hard,
        "objective_low_ambiguity": _retag(low_amb, "objective_low_ambiguity"),
        "objective_high_ambiguity": _retag(high_amb, "objective_high_ambiguity"),
        "matched_objective": _retag(matched_obj, "matched_objective"),
        "matched_subjective": _retag(matched_subj, "matched_subjective"),
    }


def _time_grid(max_time_sec: float, num_points: int) -> np.ndarray:
    num = max(int(num_points), 2)
    return np.linspace(0.0, float(max_time_sec), num=num)


def _single_problem_state(problem: Problem, time_spent: float, confidence: float, total_time_sec: float) -> ExamState:
    progress = ProblemProgress(status=ProblemStatus.IN_PROGRESS, time_spent_sec=float(time_spent))
    progress.sync_from_scalar(problem, float(confidence))
    state = ExamState(
        remaining_time_sec=max(float(total_time_sec) - float(time_spent), 0.0),
        current_problem_idx=0,
        progress=[progress],
        total_score=float(problem.score) * float(confidence),
        step_count=0,
        visit_order=[0],
        same_problem_streak=0,
    )
    state.total_score = expected_utility(state, [problem])
    return state


def _curve_bundle(
    case: ProblemCase,
    student: Any,
    dynamics_cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
    time_grid: np.ndarray,
    action_unit_sec: float,
    max_time_sec: float,
) -> dict[str, Any]:
    probabilities = np.asarray(
        [confidence_curve(case.problem, student, float(t), dynamics_cfg) for t in time_grid],
        dtype=float,
    )
    expected_scores = probabilities * float(case.problem.score)
    monotone = bool(np.all(np.diff(probabilities) >= -1e-9))

    reward_times = np.arange(0.0, float(max_time_sec), float(action_unit_sec), dtype=float)
    reward_rows: list[dict[str, float]] = []
    max_reward_delta_gap = 0.0
    for t in reward_times:
        next_t = min(t + float(action_unit_sec), float(max_time_sec))
        prev_conf = confidence_curve(case.problem, student, float(t), dynamics_cfg)
        next_conf = confidence_curve(case.problem, student, float(next_t), dynamics_cfg)
        prev_state = _single_problem_state(case.problem, float(t), prev_conf, float(max_time_sec))
        next_state = _single_problem_state(case.problem, float(next_t), next_conf, float(max_time_sec))
        delta_score = expected_utility(next_state, [case.problem]) - expected_utility(prev_state, [case.problem])
        reward = compute_step_reward(prev_state, next_state, [case.problem], "solve_more", reward_cfg)
        gap = abs(float(reward) - float(delta_score))
        max_reward_delta_gap = max(max_reward_delta_gap, gap)
        reward_rows.append(
            {
                "time_sec": float(t),
                "next_time_sec": float(next_t),
                "reward": float(reward),
                "delta_expected_score": float(delta_score),
                "marginal_gain_per_sec": float(delta_score / max(next_t - t, 1e-9)),
            }
        )

    return {
        "problem": case.describe(),
        "time_sec": [float(x) for x in time_grid],
        "probability": [float(x) for x in probabilities],
        "expected_score": [float(x) for x in expected_scores],
        "probability_floor": float(guessing_prob(case.problem, dynamics_cfg)),
        "probability_start": float(probabilities[0]),
        "probability_end": float(probabilities[-1]),
        "expected_score_start": float(expected_scores[0]),
        "expected_score_end": float(expected_scores[-1]),
        "monotone_probability": monotone,
        "reward_consistency_max_abs_error": float(max_reward_delta_gap),
        "reward_steps": reward_rows,
    }


def _write_curve_csv(curves: dict[str, dict[str, Any]], path: str) -> str:
    _ensure_dir(os.path.dirname(path) or ".")
    fieldnames = [
        "tag",
        "time_sec",
        "probability",
        "expected_score",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tag, bundle in curves.items():
            for t, p, s in zip(bundle["time_sec"], bundle["probability"], bundle["expected_score"]):
                writer.writerow(
                    {
                        "tag": tag,
                        "time_sec": float(t),
                        "probability": float(p),
                        "expected_score": float(s),
                    }
                )
    return path


def _plot_curve_group(
    curves: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    y_key: str,
    title: str,
    ylabel: str,
    save_path: str,
) -> str:
    _ensure_matplotlib()
    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag in tags:
        bundle = curves[tag]
        meta = bundle["problem"]
        label = f"{tag} | pid={meta['pid']} | {os.path.basename(meta['exam_path'])}"
        ax.plot(bundle["time_sec"], bundle[y_key], linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time Spent (sec)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _plot_marginal_gain_group(
    curves: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    title: str,
    save_path: str,
) -> str:
    _ensure_matplotlib()
    _ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag in tags:
        bundle = curves[tag]
        meta = bundle["problem"]
        label = f"{tag} | pid={meta['pid']} | {os.path.basename(meta['exam_path'])}"
        xs = [row["next_time_sec"] for row in bundle["reward_steps"]]
        ys = [row["marginal_gain_per_sec"] for row in bundle["reward_steps"]]
        ax.plot(xs, ys, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time Spent After Step (sec)")
    ax.set_ylabel("Marginal Expected Score Gain / sec")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def _sample_at_times(bundle: dict[str, Any], times: list[float]) -> list[dict[str, float]]:
    xs = np.asarray(bundle["time_sec"], dtype=float)
    ps = np.asarray(bundle["probability"], dtype=float)
    ss = np.asarray(bundle["expected_score"], dtype=float)
    out = []
    for t in times:
        idx = int(np.argmin(np.abs(xs - float(t))))
        out.append(
            {
                "time_sec": float(xs[idx]),
                "probability": float(ps[idx]),
                "expected_score": float(ss[idx]),
            }
        )
    return out


def _build_summary(
    curves: dict[str, dict[str, Any]],
    *,
    time_snapshots: list[float],
    max_time_sec: float,
) -> dict[str, Any]:
    easy_obj = curves["objective_easy"]
    hard_obj = curves["objective_hard"]
    easy_subj = curves["subjective_easy"]
    hard_subj = curves["subjective_hard"]
    low_amb = curves["objective_low_ambiguity"]
    high_amb = curves["objective_high_ambiguity"]
    matched_obj = curves["matched_objective"]
    matched_subj = curves["matched_subjective"]

    def _at(bundle: dict[str, Any], target_time: float) -> float:
        xs = np.asarray(bundle["time_sec"], dtype=float)
        ys = np.asarray(bundle["probability"], dtype=float)
        idx = int(np.argmin(np.abs(xs - float(target_time))))
        return float(ys[idx])

    selected_times = [t for t in time_snapshots if t <= max_time_sec]
    mid_time = selected_times[min(len(selected_times) - 1, 2)] if selected_times else max_time_sec / 2.0

    ceiling_values = {tag: float(bundle["probability_end"]) for tag, bundle in curves.items()}
    reward_errors = {tag: float(bundle["reward_consistency_max_abs_error"]) for tag, bundle in curves.items()}

    return {
        "easy_vs_hard_objective": {
            "comparison_time_sec": float(mid_time),
            "easy_probability": _at(easy_obj, mid_time),
            "hard_probability": _at(hard_obj, mid_time),
            "easy_gt_hard": bool(_at(easy_obj, mid_time) > _at(hard_obj, mid_time)),
        },
        "easy_vs_hard_subjective": {
            "comparison_time_sec": float(mid_time),
            "easy_probability": _at(easy_subj, mid_time),
            "hard_probability": _at(hard_subj, mid_time),
            "easy_gt_hard": bool(_at(easy_subj, mid_time) > _at(hard_subj, mid_time)),
        },
        "objective_vs_subjective_matched": {
            "comparison_time_sec": float(mid_time),
            "objective_probability": _at(matched_obj, mid_time),
            "subjective_probability": _at(matched_subj, mid_time),
            "objective_floor": float(matched_obj["probability_floor"]),
            "subjective_floor": float(matched_subj["probability_floor"]),
            "objective_start_probability": float(matched_obj["probability_start"]),
            "subjective_start_probability": float(matched_subj["probability_start"]),
        },
        "ambiguity_effect": {
            "comparison_time_sec": float(mid_time),
            "low_ambiguity_probability": _at(low_amb, mid_time),
            "high_ambiguity_probability": _at(high_amb, mid_time),
            "low_ambiguity_gt_high_ambiguity": bool(_at(low_amb, mid_time) > _at(high_amb, mid_time)),
        },
        "ceiling_check": {
            "max_time_sec": float(max_time_sec),
            "final_probabilities": ceiling_values,
            "all_below_0995": bool(all(v < 0.995 for v in ceiling_values.values())),
        },
        "monotonicity_check": {
            tag: bool(bundle["monotone_probability"]) for tag, bundle in curves.items()
        },
        "reward_consistency_check": {
            "max_abs_error_by_case": reward_errors,
            "global_max_abs_error": float(max(reward_errors.values()) if reward_errors else 0.0),
        },
        "snapshots": {
            tag: _sample_at_times(bundle, selected_times)
            for tag, bundle in curves.items()
        },
    }


def _write_report(
    path: str,
    *,
    config_path: str,
    student_desc: str,
    action_unit_sec: float,
    max_time_sec: float,
    curves: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    plot_paths: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# Environment Validation Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Student: `{student_desc}`")
    lines.append(f"- Action time unit: `{action_unit_sec}` sec")
    lines.append(f"- Max curve horizon: `{max_time_sec}` sec")
    lines.append("")
    lines.append("## Representative Problems")
    lines.append("")
    for tag, bundle in curves.items():
        meta = bundle["problem"]
        lines.append(
            f"- `{tag}`: exam=`{os.path.basename(meta['exam_path'])}`, pid=`{meta['pid']}`, "
            f"type=`{meta['problem_type']}`, score=`{meta['score']}`, "
            f"anchor=`{meta['anchor']:.3f}`, entropy=`{meta['choice_entropy']:.3f}`"
        )
    lines.append("")
    lines.append("## Key Checks")
    lines.append("")
    easy_obj = summary["easy_vs_hard_objective"]
    lines.append(
        f"- Easy vs hard objective at `{easy_obj['comparison_time_sec']:.1f}`s: "
        f"`{easy_obj['easy_probability']:.4f}` vs `{easy_obj['hard_probability']:.4f}` "
        f"(easy > hard = `{easy_obj['easy_gt_hard']}`)"
    )
    easy_subj = summary["easy_vs_hard_subjective"]
    lines.append(
        f"- Easy vs hard subjective at `{easy_subj['comparison_time_sec']:.1f}`s: "
        f"`{easy_subj['easy_probability']:.4f}` vs `{easy_subj['hard_probability']:.4f}` "
        f"(easy > hard = `{easy_subj['easy_gt_hard']}`)"
    )
    obj_subj = summary["objective_vs_subjective_matched"]
    lines.append(
        f"- Matched objective vs subjective at `{obj_subj['comparison_time_sec']:.1f}`s: "
        f"objective=`{obj_subj['objective_probability']:.4f}`, subjective=`{obj_subj['subjective_probability']:.4f}`, "
        f"floors=`{obj_subj['objective_floor']:.4f}` / `{obj_subj['subjective_floor']:.4f}`, "
        f"t=0 start=`{obj_subj['objective_start_probability']:.4f}` / `{obj_subj['subjective_start_probability']:.4f}`"
    )
    ambiguity = summary["ambiguity_effect"]
    lines.append(
        f"- Low vs high ambiguity objective at `{ambiguity['comparison_time_sec']:.1f}`s: "
        f"`{ambiguity['low_ambiguity_probability']:.4f}` vs `{ambiguity['high_ambiguity_probability']:.4f}` "
        f"(low ambiguity > high ambiguity = `{ambiguity['low_ambiguity_gt_high_ambiguity']}`)"
    )
    ceiling = summary["ceiling_check"]
    lines.append(
        f"- Final probability ceiling at `{ceiling['max_time_sec']:.1f}`s stays below `0.995` for all representative cases: "
        f"`{ceiling['all_below_0995']}`"
    )
    reward = summary["reward_consistency_check"]
    lines.append(
        f"- Reward consistency max abs error (`step reward` vs `delta expected score`): "
        f"`{reward['global_max_abs_error']:.10f}`"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for name, plot_path in plot_paths.items():
        lines.append(f"- `{name}`: `{plot_path}`")
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dynamics and reward curves without RL.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--exam-data", type=str, default=None)
    parser.add_argument("--student-data", type=str, default=None)
    parser.add_argument("--student-id", type=str, default=None)
    parser.add_argument("--student-level", type=str, default=None)
    parser.add_argument("--exam-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-time-sec", type=float, default=None)
    parser.add_argument("--num-points", type=int, default=181)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    env, student = _load_env_and_student(config, args)

    dynamics_cfg = dict(config.get("dynamics", {}))
    reward_cfg = dict(config.get("reward", {}))
    max_time_sec = float(args.max_time_sec if args.max_time_sec is not None else config.get("exam", {}).get("total_time_sec", 6000))
    action_unit_sec = float(config.get("exam", {}).get("action_time_unit_sec", 30.0))
    time_grid = _time_grid(max_time_sec, args.num_points)

    raw_cases = _all_problem_cases(env, dynamics_cfg)
    selected = _selected_cases(raw_cases)
    curves = {
        tag: _curve_bundle(
            case=case,
            student=student,
            dynamics_cfg=dynamics_cfg,
            reward_cfg=reward_cfg,
            time_grid=time_grid,
            action_unit_sec=action_unit_sec,
            max_time_sec=max_time_sec,
        )
        for tag, case in selected.items()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("runs", f"env_validation_{timestamp}")
    plot_dir = os.path.join(output_dir, "plots")
    _ensure_dir(plot_dir)

    plot_paths = {
        "objective_probability": _plot_curve_group(
            curves,
            ["objective_easy", "objective_hard"],
            y_key="probability",
            title="Objective Problems: Easy vs Hard Probability Curve",
            ylabel="P(correct | item, student, time)",
            save_path=os.path.join(plot_dir, "objective_probability.png"),
        ),
        "objective_expected_score": _plot_curve_group(
            curves,
            ["objective_easy", "objective_hard"],
            y_key="expected_score",
            title="Objective Problems: Easy vs Hard Expected Score Curve",
            ylabel="Expected Score",
            save_path=os.path.join(plot_dir, "objective_expected_score.png"),
        ),
        "subjective_probability": _plot_curve_group(
            curves,
            ["subjective_easy", "subjective_hard"],
            y_key="probability",
            title="Subjective Problems: Easy vs Hard Probability Curve",
            ylabel="P(correct | item, student, time)",
            save_path=os.path.join(plot_dir, "subjective_probability.png"),
        ),
        "subjective_expected_score": _plot_curve_group(
            curves,
            ["subjective_easy", "subjective_hard"],
            y_key="expected_score",
            title="Subjective Problems: Easy vs Hard Expected Score Curve",
            ylabel="Expected Score",
            save_path=os.path.join(plot_dir, "subjective_expected_score.png"),
        ),
        "cross_type_probability": _plot_curve_group(
            curves,
            ["matched_objective", "matched_subjective"],
            y_key="probability",
            title="Matched Objective vs Subjective Probability Curve",
            ylabel="P(correct | item, student, time)",
            save_path=os.path.join(plot_dir, "cross_type_probability.png"),
        ),
        "cross_type_expected_score": _plot_curve_group(
            curves,
            ["matched_objective", "matched_subjective"],
            y_key="expected_score",
            title="Matched Objective vs Subjective Expected Score Curve",
            ylabel="Expected Score",
            save_path=os.path.join(plot_dir, "cross_type_expected_score.png"),
        ),
        "ambiguity_probability": _plot_curve_group(
            curves,
            ["objective_low_ambiguity", "objective_high_ambiguity"],
            y_key="probability",
            title="Objective Problems: Low vs High Ambiguity Probability Curve",
            ylabel="P(correct | item, student, time)",
            save_path=os.path.join(plot_dir, "ambiguity_probability.png"),
        ),
        "ambiguity_marginal_gain": _plot_marginal_gain_group(
            curves,
            ["objective_low_ambiguity", "objective_high_ambiguity"],
            title="Objective Problems: Low vs High Ambiguity Marginal Gain",
            save_path=os.path.join(plot_dir, "ambiguity_marginal_gain.png"),
        ),
    }

    summary = _build_summary(
        curves,
        time_snapshots=[30.0, 60.0, 120.0, 300.0, 600.0, max_time_sec],
        max_time_sec=max_time_sec,
    )
    curve_csv = _write_curve_csv(curves, os.path.join(output_dir, "curve_data.csv"))
    curve_json = os.path.join(output_dir, "curve_data.json")
    save_json(curves, curve_json, indent=2)
    summary_json = os.path.join(output_dir, "summary.json")
    save_json(
        {
            "config_path": args.config,
            "student_id": getattr(student, "student_id", None),
            "dynamics": dynamics_cfg,
            "reward": reward_cfg,
            "summary": summary,
            "artifacts": {
                "curve_csv": curve_csv,
                "curve_json": curve_json,
                "plots": plot_paths,
            },
        },
        summary_json,
        indent=2,
    )
    report_path = _write_report(
        os.path.join(output_dir, "report.md"),
        config_path=args.config,
        student_desc=getattr(student, "student_id", "unknown"),
        action_unit_sec=action_unit_sec,
        max_time_sec=max_time_sec,
        curves=curves,
        summary=summary,
        plot_paths=plot_paths,
    )

    print(
        {
            "output_dir": output_dir,
            "summary_json": summary_json,
            "report_md": report_path,
            "curve_csv": curve_csv,
            "plots": plot_paths,
        }
    )


if __name__ == "__main__":
    main()
