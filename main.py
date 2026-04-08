from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

from analysis.evaluator import evaluate_heuristics_table, evaluate_policy, save_results_json, save_table_csv
from agents.train_rl import evaluate_trained_model, train_from_config
from utils.io import load_config
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


def _load_trained_model(model_path: str, algorithm: str):
    algo = algorithm.lower()
    if algo == "ppo":
        if PPO is None:
            raise ImportError("stable-baselines3 is required to load PPO model.")
        return PPO.load(model_path)
    if algo == "dqn":
        if DQN is None:
            raise ImportError("stable-baselines3 is required to load DQN model.")
        return DQN.load(model_path)
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
    model = _load_trained_model(model_path=model_path, algorithm=algorithm)
    summary = evaluate_trained_model(
        model=model,
        config=config,
        n_episodes=episodes,
        algorithm=algorithm,
        seed=seed,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exam RL project entrypoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "heuristic"], required=True)
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
