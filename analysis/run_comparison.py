"""Comparison table: heuristic baselines vs trained RL models.

Usage:
    # Heuristics only (no trained model needed)
    python analysis/run_comparison.py --config configs/default.yaml

    # Include trained RL models found under a runs/ directory
    python analysis/run_comparison.py --config configs/default.yaml --runs-dir runs/

    # Target a single run directory
    python analysis/run_comparison.py --config configs/default.yaml --runs-dir runs/ppo_20260409_100407

    # Full options
    python analysis/run_comparison.py \\
        --config configs/default.yaml \\
        --runs-dir runs/ \\
        --episodes 60 \\
        --realized-rollouts 100 \\
        --output results/

Output:
    results/comparison_table.csv   — machine-readable table
    Formatted table printed to stdout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from agents.heuristic_agents import HEURISTIC_POLICIES
from analysis.evaluator import evaluate_policy
from utils.io import load_config, save_results_csv


# ---------------------------------------------------------------------------
# RL model discovery
# ---------------------------------------------------------------------------

def _find_rl_models(runs_dir: str) -> list[dict]:
    """Scan runs_dir for final model checkpoints.

    Looks for patterns:
        <runs_dir>/**/checkpoints/*_final.zip
        <runs_dir>/**/checkpoints/ppo_final.zip
        <runs_dir>/**/checkpoints/dqn_final.zip

    Returns a list of dicts with keys: path (without .zip), algorithm, run_name.
    """
    if not os.path.isdir(runs_dir):
        return []

    found: list[dict] = []
    # Walk up to 3 levels deep to find checkpoints/
    for root, dirs, files in os.walk(runs_dir):
        depth = root.replace(runs_dir, "").count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        for fname in files:
            if not fname.endswith("_final.zip"):
                continue
            zip_path = os.path.join(root, fname)
            model_path = zip_path[:-4]  # strip .zip — SB3 loads without extension
            stem = fname[:-4]            # e.g. "ppo_final"
            algo = stem.split("_")[0].lower()  # "ppo" or "dqn"
            run_name = os.path.basename(os.path.dirname(os.path.dirname(zip_path)))
            found.append({"path": model_path, "algorithm": algo, "run_name": run_name})
    return found


def _load_rl_model(model_path: str, algorithm: str):
    """Load a stable-baselines3 model from disk."""
    try:
        if algorithm == "ppo":
            from stable_baselines3 import PPO
            return PPO.load(model_path)
        if algorithm == "dqn":
            from stable_baselines3 import DQN
            return DQN.load(model_path)
    except Exception as e:
        print(f"  [warn] Failed to load {model_path}: {e}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ("policy",                "Policy",              "<25"),
    ("type",                  "Type",                "<10"),
    ("mean_expected_score",   "Exp.Score",           ">10.3f"),
    ("mean_realized_score",   "Real.Score",          ">11.3f"),
    ("mean_realized_score_std", "Real.Std",          ">9.3f"),
    ("mean_reward",           "Reward",              ">9.3f"),
    ("mean_coverage_fraction","Coverage",            ">9.3f"),
    ("mean_solved_count",     "Solved",              ">7.2f"),
    ("mean_steps",            "Steps",               ">7.1f"),
]


def _format_table(rows: list[dict]) -> str:
    """Return a fixed-width plain-text comparison table."""
    header_parts = []
    sep_parts = []
    for key, label, fmt in _TABLE_COLS:
        width = int(fmt.strip("<>f").split(".")[0])
        header_parts.append(f"{label:{fmt.replace('f', 's').replace('.3', '').replace('.2','').replace('.1','')[:3]}}")
        sep_parts.append("-" * max(width, len(label)))

    # Build header using label widths
    col_widths = []
    for key, label, fmt in _TABLE_COLS:
        spec = fmt.lstrip("<>")
        width = int(spec.split(".")[0]) if spec else len(label)
        col_widths.append(max(width, len(label)))

    def _fmt_cell(value, fmt: str, width: int) -> str:
        align = "<" if fmt.startswith("<") else ">"
        if isinstance(value, float):
            decimals = int(fmt.split(".")[-1].replace("f", "")) if "." in fmt else 0
            cell = f"{value:.{decimals}f}"
        else:
            cell = str(value)
        return f"{cell:{align}{width}}"

    header = "  ".join(
        f"{label:{('<' if fmt.startswith('<') else '>')}{col_widths[i]}}"
        for i, (_, label, fmt) in enumerate(_TABLE_COLS)
    )
    sep = "  ".join("-" * w for w in col_widths)
    lines = [header, sep]
    for row in rows:
        parts = []
        for i, (key, label, fmt) in enumerate(_TABLE_COLS):
            val = row.get(key, "")
            parts.append(_fmt_cell(val, fmt, col_widths[i]))
        lines.append("  ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_comparison(
    config: dict,
    runs_dir: str | None = None,
    episodes: int = 50,
    realized_rollouts: int = 100,
    output_dir: str = "results",
    student_level: str | None = None,
) -> list[dict]:
    """Evaluate all heuristics + any RL models found in runs_dir.

    Returns list of result row dicts (one per policy).
    """
    rows: list[dict] = []

    # ── 1. Heuristic baselines ──────────────────────────────────────────────
    print(f"\nEvaluating {len(HEURISTIC_POLICIES)} heuristic baselines  "
          f"({episodes} episodes, {realized_rollouts} realized rollouts each)…")
    for policy_name in HEURISTIC_POLICIES:
        print(f"  {policy_name}…", end=" ", flush=True)
        result = evaluate_policy(
            config=config,
            policy_name=policy_name,
            episodes=episodes,
            student_level=student_level,
            seed=42,
            realized_rollouts=realized_rollouts,
        )
        s = result["summary"]
        row = {
            "policy": policy_name,
            "type": "heuristic",
            "mean_expected_score":    round(s["mean_score"], 4),
            "mean_realized_score":    round(s["mean_realized_score"], 4),
            "mean_realized_score_std": round(s["mean_realized_score_std"], 4),
            "mean_reward":            round(s["mean_reward"], 4),
            "mean_coverage_fraction": round(s["mean_coverage_fraction"], 4),
            "mean_solved_count":      round(s["mean_solved_count"], 4),
            "mean_steps":             round(s["mean_steps"], 4),
        }
        rows.append(row)
        print(f"exp={row['mean_expected_score']:.3f}  real={row['mean_realized_score']:.3f}")

    # ── 2. RL models ────────────────────────────────────────────────────────
    if runs_dir:
        models = _find_rl_models(runs_dir)
        if not models:
            print(f"\n[info] No *_final.zip models found under {runs_dir}")
        else:
            print(f"\nEvaluating {len(models)} RL model(s) from {runs_dir}…")
        for m in models:
            label = f"{m['algorithm'].upper()} ({m['run_name']})"
            print(f"  {label}…", end=" ", flush=True)
            model = _load_rl_model(m["path"], m["algorithm"])
            if model is None:
                print("SKIP (load failed)")
                continue
            result = evaluate_policy(
                config=config,
                policy_name=f"rl_{m['algorithm']}",
                episodes=episodes,
                student_level=student_level,
                rl_model=model,
                rl_algorithm=m["algorithm"],
                seed=42,
                realized_rollouts=realized_rollouts,
            )
            s = result["summary"]
            row = {
                "policy": f"rl_{m['algorithm']}_{m['run_name']}",
                "type": m["algorithm"],
                "mean_expected_score":    round(s["mean_score"], 4),
                "mean_realized_score":    round(s["mean_realized_score"], 4),
                "mean_realized_score_std": round(s["mean_realized_score_std"], 4),
                "mean_reward":            round(s["mean_reward"], 4),
                "mean_coverage_fraction": round(s["mean_coverage_fraction"], 4),
                "mean_solved_count":      round(s["mean_solved_count"], 4),
                "mean_steps":             round(s["mean_steps"], 4),
            }
            rows.append(row)
            print(f"exp={row['mean_expected_score']:.3f}  real={row['mean_realized_score']:.3f}")

    # ── 3. Sort by realized score descending ────────────────────────────────
    rows.sort(key=lambda r: r["mean_realized_score"], reverse=True)

    # ── 4. Save CSV ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    save_results_csv(rows, csv_path)
    print(f"\nCSV saved → {csv_path}")

    # ── 5. Save full JSON (includes episode-level data from last policy) ────
    json_path = os.path.join(output_dir, "comparison_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all heuristic baselines (+ optional RL models) and save a comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",  type=str, default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory to search for trained RL models (*_final.zip)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per policy (default: 50)")
    parser.add_argument("--realized-rollouts", type=int, default=100,
                        help="Bernoulli rollouts per episode for realized score (default: 100)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for CSV/JSON (default: results/)")
    parser.add_argument("--student-level", type=str, default=None,
                        choices=["low", "mid", "high"],
                        help="Fix student level for all episodes (default: mixed)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)

    rows = run_comparison(
        config=cfg,
        runs_dir=args.runs_dir,
        episodes=args.episodes,
        realized_rollouts=args.realized_rollouts,
        output_dir=args.output,
        student_level=args.student_level,
    )

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(_format_table(rows))
    print("=" * 80)
