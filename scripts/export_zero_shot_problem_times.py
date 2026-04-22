from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_HEADER = ["pid", "difficulty", "type", "score", "error_rate", "avg_time_sec"]
SUBJECT_EXAM_PATHS = {
    "calculus": Path("data/25_math_calculus.json"),
    "geometry": Path("data/25_math_geometry.json"),
    "prob_stat": Path("data/25_math_prob_stat.json"),
}
DEFAULT_SUBJECTS = tuple(SUBJECT_EXAM_PATHS)
DEFAULT_LEVELS = ("low", "mid", "high")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_error_rates(exam_path: Path) -> dict[int, Any]:
    exam = _load_json(exam_path)
    return {int(problem["pid"]): problem.get("error_rate", "") for problem in exam["problems"]}


def _latest_detailed_file(algo_root: Path, algorithm: str, subject: str, level: str) -> Path | None:
    candidates = list((algo_root / subject / level).glob(f"eval_{algorithm}_*/rl_eval_detailed.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _problem_time_rows(detailed_path: Path, error_by_pid: dict[int, Any]) -> list[list[Any]]:
    detailed = _load_json(detailed_path)
    problem_times = detailed.get("problem_avg_time_by_pid")
    if not isinstance(problem_times, list):
        raise ValueError(f"Missing problem_avg_time_by_pid in {detailed_path}")

    rows: list[list[Any]] = []
    for item in sorted(problem_times, key=lambda row: int(row["pid"])):
        pid = int(item["pid"])
        rows.append(
            [
                pid,
                item["difficulty_level"],
                item["problem_type"],
                item["score"],
                error_by_pid.get(pid, ""),
                item["avg_time_sec"],
            ]
        )
    return rows


def export_problem_times(
    *,
    algorithm: str,
    runs_root: Path,
    output_dir: Path,
    subjects: tuple[str, ...],
    levels: tuple[str, ...],
    dry_run: bool,
) -> int:
    algo_root = runs_root / algorithm
    if not algo_root.exists():
        raise FileNotFoundError(f"Zero-shot run directory not found: {algo_root}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for subject in subjects:
        exam_path = SUBJECT_EXAM_PATHS[subject]
        error_by_pid = _load_error_rates(exam_path)

        for level in levels:
            detailed_path = _latest_detailed_file(algo_root, algorithm, subject, level)
            if detailed_path is None:
                print(f"[skip] missing eval result: {algorithm}/{subject}/{level}")
                continue

            rows = _problem_time_rows(detailed_path, error_by_pid)
            output_path = output_dir / f"{subject}_{level}.csv"

            if dry_run:
                print(f"[dry-run] {output_path} <- {detailed_path}")
                saved += 1
                continue

            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)

            print(f"[saved] {output_path} <- {detailed_path}")
            saved += 1

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-problem average time CSVs from zero-shot PPO/DQN evaluation results.",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        choices=("ppo", "dqn"),
        required=True,
        help="Algorithm result directory to export from.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/zero_shot"),
        help="Root directory containing zero-shot results. Default: runs/zero_shot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: results/<algorithm>_zero_shot_problem_times",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=DEFAULT_SUBJECTS,
        default=DEFAULT_SUBJECTS,
        help="Subjects to export. Default: all subjects.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=DEFAULT_LEVELS,
        default=DEFAULT_LEVELS,
        help="Student levels to export. Default: all levels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without writing CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or Path("results") / f"{args.algorithm}_zero_shot_problem_times"
    saved = export_problem_times(
        algorithm=args.algorithm,
        runs_root=args.runs_root,
        output_dir=output_dir,
        subjects=tuple(args.subjects),
        levels=tuple(args.levels),
        dry_run=bool(args.dry_run),
    )
    print(f"[done] exported {saved} CSV file(s)")


if __name__ == "__main__":
    main()
