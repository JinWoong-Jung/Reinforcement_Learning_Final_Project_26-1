from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import datetime
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "rlproject_matplotlib"))

import numpy as np

from analysis.evaluator import evaluate_heuristics_table, evaluate_policy, save_results_json, save_table_csv
from agents.train_rl import _build_env, evaluate_trained_model, train_from_config
from utils.io import load_config
from utils.model_compat import build_sb3_custom_objects, install_numpy_pickle_compat
from utils.seed import set_global_seed

try:
    from stable_baselines3 import DQN, PPO
except ImportError:  # pragma: no cover
    DQN = None
    PPO = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_obs_stats_path(model_path: str) -> str | None:
    model_dir = os.path.dirname(model_path)
    candidate = os.path.join(model_dir, "obs_stats.npz")
    if os.path.exists(candidate):
        return candidate
    stem_candidate = os.path.join(os.path.splitext(model_path)[0], "obs_stats.npz")
    if os.path.exists(stem_candidate):
        return stem_candidate
    return None


def _load_trained_model(model_path: str, algorithm: str, config: dict[str, Any]):
    algo = algorithm.lower()
    install_numpy_pickle_compat()
    seed = int(config.get("experiment", {}).get("seed", 42))
    env = _build_env(config=config, for_dqn=algo == "dqn", seed=seed)
    custom_objects = build_sb3_custom_objects(config, algo, env)
    if algo == "ppo":
        if PPO is None:
            raise ImportError("stable-baselines3 is required to load PPO model.")
        return PPO.load(model_path, env=env, custom_objects=custom_objects)
    if algo == "dqn":
        if DQN is None:
            raise ImportError("stable-baselines3 is required to load DQN model.")
        return DQN.load(model_path, env=env, custom_objects=custom_objects)
    raise ValueError("algorithm must be 'ppo' or 'dqn'")


def run_train(config: dict[str, Any], output_root: str) -> dict[str, Any]:
    result = train_from_config(config=config, output_root=output_root)
    return {
        "mode": "train",
        "final_model_path": result["final_model_path"],
        "eval": result["eval"],
        "paths": result["paths"],
    }


def run_heuristic(config: dict[str, Any], output_root: str, episodes: int, seed: int) -> dict[str, Any]:
    student_cfg = dict(config.get("student", {}))
    rows = evaluate_heuristics_table(
        config=config,
        episodes=episodes,
        seed=seed,
        student_id=student_cfg.get("fixed_id"),
        student_level=student_cfg.get("fixed_level"),
    )
    out_dir = os.path.join(output_root, f"heuristic_{_timestamp()}")
    _ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "heuristic_table.csv")
    json_path = os.path.join(out_dir, "heuristic_table.json")
    save_table_csv(rows, csv_path)
    save_results_json({"rows": rows, "episodes": episodes}, json_path)

    return {
        "mode": "heuristic",
        "episodes": episodes,
        "csv_path": csv_path,
        "json_path": json_path,
        "rows": rows,
    }


def run_eval(
    config: dict[str, Any],
    output_root: str,
    model_path: str,
    algorithm: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    student_cfg = dict(config.get("student", {}))
    obs_stats_path = _resolve_obs_stats_path(model_path)
    model = _load_trained_model(model_path=model_path, algorithm=algorithm, config=config)
    summary = evaluate_trained_model(
        model=model,
        config=config,
        n_episodes=episodes,
        algorithm=algorithm,
        seed=seed,
        obs_stats_path=obs_stats_path,
    )
    detailed = evaluate_policy(
        config=config,
        policy_name=f"rl_{algorithm}",
        episodes=episodes,
        student_id=student_cfg.get("fixed_id"),
        student_level=student_cfg.get("fixed_level"),
        rl_model=model,
        rl_algorithm=algorithm,
        seed=seed,
        obs_stats_path=obs_stats_path,
    )

    out_dir = os.path.join(output_root, f"eval_{algorithm}_{_timestamp()}")
    _ensure_dir(out_dir)
    summary_path = os.path.join(out_dir, "rl_eval_summary.json")
    detailed_path = os.path.join(out_dir, "rl_eval_detailed.json")
    save_results_json(summary, summary_path)
    save_results_json(detailed, detailed_path)

    return {
        "mode": "eval",
        "algorithm": algorithm,
        "model_path": model_path,
        "summary_path": summary_path,
        "detailed_path": detailed_path,
        "summary": summary,
    }


def _exam_label(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{parent}_{stem}" if parent else stem


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def run_cross_validation(
    config: dict[str, Any],
    output_root: str,
    episodes: int,
    seed: int,
    train_final: bool = True,
) -> dict[str, Any]:
    """Run leave-one-exam-out CV and then train one final model on all exams."""
    data_cfg = dict(config.get("data", {}))
    exam_paths = list(data_cfg.get("exam_paths", []) or [])
    if len(exam_paths) < 2:
        raise ValueError("CV mode requires data.exam_paths with at least two exam JSON files.")

    algo = str(config.get("training", {}).get("algorithm", "ppo")).lower()
    timestamp = _timestamp()
    cv_root = os.path.join(output_root, f"cv_{algo}_{timestamp}")
    folds_root = os.path.join(cv_root, "folds")
    eval_root = os.path.join(cv_root, "heldout_eval")
    final_root = os.path.join(cv_root, "final")
    _ensure_dir(folds_root)
    _ensure_dir(eval_root)

    fold_results: list[dict[str, Any]] = []
    for fold_idx, heldout_path in enumerate(exam_paths, start=1):
        heldout_label = _exam_label(str(heldout_path))
        fold_cfg = copy.deepcopy(config)
        fold_cfg.setdefault("data", {})
        fold_cfg["data"]["exam_paths"] = [p for p in exam_paths if p != heldout_path]
        fold_cfg["data"].pop("exam_path", None)
        fold_cfg.setdefault("experiment", {})
        fold_cfg["experiment"]["seed"] = int(seed)
        fold_cfg["experiment"]["name"] = f"{fold_cfg['experiment'].get('name', algo)}_fold_{heldout_label}"

        run_name = f"fold_{fold_idx:02d}_holdout_{heldout_label}"
        print(f"\n[cv] === fold {fold_idx}/{len(exam_paths)} heldout={heldout_path} ===")
        train_result = train_from_config(config=fold_cfg, output_root=folds_root, run_name=run_name)

        eval_cfg = copy.deepcopy(fold_cfg)
        eval_cfg.setdefault("data", {})
        eval_cfg["data"]["exam_path"] = heldout_path
        eval_cfg["data"].pop("exam_paths", None)
        obs_stats_path = _resolve_obs_stats_path(train_result["final_model_path"])
        model = _load_trained_model(
            model_path=train_result["final_model_path"],
            algorithm=algo,
            config=eval_cfg,
        )
        heldout_summary = evaluate_trained_model(
            model=model,
            config=eval_cfg,
            n_episodes=episodes,
            algorithm=algo,
            seed=seed,
            obs_stats_path=obs_stats_path,
        )

        fold_record = {
            "fold": fold_idx,
            "heldout_exam_path": heldout_path,
            "train_exam_paths": fold_cfg["data"]["exam_paths"],
            "run_dir": train_result["paths"]["base"],
            "final_model_path": train_result["final_model_path"],
            "train_eval": train_result["eval"],
            "heldout_eval": heldout_summary,
        }
        fold_results.append(fold_record)
        fold_eval_path = os.path.join(eval_root, f"fold_{fold_idx:02d}_{heldout_label}.json")
        save_results_json(fold_record, fold_eval_path)
        print(
            "[cv] heldout "
            f"score={heldout_summary['mean_score']:.4f} "
            f"reward={heldout_summary['mean_reward']:.4f} "
            f"coverage={heldout_summary['mean_coverage_fraction']:.4f}"
        )

    heldout_scores = [float(r["heldout_eval"]["mean_score"]) for r in fold_results]
    heldout_rewards = [float(r["heldout_eval"]["mean_reward"]) for r in fold_results]
    heldout_coverages = [float(r["heldout_eval"]["mean_coverage_fraction"]) for r in fold_results]

    final_result = None
    if train_final:
        final_cfg = copy.deepcopy(config)
        final_cfg.setdefault("data", {})
        final_cfg["data"]["exam_paths"] = list(exam_paths)
        final_cfg["data"].pop("exam_path", None)
        final_cfg.setdefault("experiment", {})
        final_cfg["experiment"]["seed"] = int(seed)
        final_cfg["experiment"]["name"] = f"{final_cfg['experiment'].get('name', algo)}_final_all"
        print(f"\n[cv] === final training on all {len(exam_paths)} exams ===")
        final_result = train_from_config(config=final_cfg, output_root=final_root, run_name="final_all")

    summary = {
        "mode": "cv",
        "algorithm": algo,
        "episodes": episodes,
        "exam_paths": exam_paths,
        "cv_root": cv_root,
        "mean_heldout_score": _mean(heldout_scores),
        "std_heldout_score": _std(heldout_scores),
        "mean_heldout_reward": _mean(heldout_rewards),
        "std_heldout_reward": _std(heldout_rewards),
        "mean_heldout_coverage_fraction": _mean(heldout_coverages),
        "folds": fold_results,
        "final": final_result,
    }
    summary_path = os.path.join(cv_root, "cv_summary.json")
    summary["summary_path"] = summary_path
    save_results_json(summary, summary_path)
    print(f"\n[cv] summary={summary_path}")
    print(f"[cv] mean_heldout_score={summary['mean_heldout_score']:.4f} ± {summary['std_heldout_score']:.4f}")
    if final_result is not None:
        print(f"[cv] final_model={final_result['final_model_path']}.zip")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exam RL project entrypoint")
    parser.add_argument("--config", type=str, default="configs/ppo/geometry/mid.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "heuristic", "cv"], required=True)
    parser.add_argument("--output", type=str, default="runs")
    parser.add_argument("--exam-data", type=str, default=None, help="Exam JSON path (overrides config)")
    parser.add_argument("--student-data", type=str, default=None, help="Student JSON path (overrides config)")
    parser.add_argument("--student-id", type=str, default=None, help="Fixed student id for all episodes")
    parser.add_argument(
        "--student-level",
        type=str,
        choices=["low", "mid", "high"],
        default=None,
        help="Fixed synthetic student level for all episodes",
    )
    parser.add_argument(
        "--student-preset",
        type=str,
        default=None,
        help="Student level preset JSON path (overrides config)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override eval/heuristic episode count")
    parser.add_argument("--model-path", type=str, default=None, help="Required for --mode eval")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "dqn"], default=None, help="Used in eval mode")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-final",
        action="store_true",
        help="In --mode cv, run folds only and skip final training on all exams.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("data", {})
    if args.exam_data:
        config["data"]["exam_path"] = args.exam_data
        config["data"].pop("exam_paths", None)
    if args.student_data:
        config["data"]["student_path"] = args.student_data
    if args.student_preset:
        config["data"]["student_preset_path"] = args.student_preset
    config.setdefault("student", {})
    if args.student_id:
        config["student"]["fixed_id"] = args.student_id
    if args.student_level:
        config["student"]["fixed_level"] = args.student_level

    seed = int(args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 42))
    set_global_seed(seed)

    train_cfg = config.get("training", {})
    default_eval_episodes = int(train_cfg.get("eval_episodes", 100))
    episodes = int(args.episodes) if args.episodes is not None else default_eval_episodes

    if args.mode == "train":
        result = run_train(config=config, output_root=args.output)
    elif args.mode == "heuristic":
        result = run_heuristic(config=config, output_root=args.output, episodes=episodes, seed=seed)
    elif args.mode == "cv":
        if args.exam_data:
            raise ValueError("--mode cv uses data.exam_paths from the config; do not pass --exam-data.")
        result = run_cross_validation(
            config=config,
            output_root=args.output,
            episodes=episodes,
            seed=seed,
            train_final=not args.no_final,
        )
    else:
        model_path = args.model_path
        if not model_path:
            raise ValueError("--model-path is required when --mode eval")
        algorithm = args.algorithm or str(train_cfg.get("algorithm", "ppo")).lower()
        result = run_eval(
            config=config,
            output_root=args.output,
            model_path=model_path,
            algorithm=algorithm,
            episodes=episodes,
            seed=seed,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
