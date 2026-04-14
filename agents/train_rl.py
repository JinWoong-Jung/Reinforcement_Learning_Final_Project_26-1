from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from env.exam_env import ExamStrategyEnv
from env.state import solved_criteria_from_config
from utils.io import load_config

HAS_GYM = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    HAS_GYM = False
    gym = None
    spaces = None

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover
    PPO = None
    DQN = None
    BaseCallback = None
    CallbackList = None
    CheckpointCallback = None
    DummyVecEnv = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class DiscreteActionWrapper(gym.Wrapper if gym is not None else object):
    """Converts MultiDiscrete action spaces into a single Discrete space for DQN."""

    def __init__(self, env):
        if gym is not None:
            super().__init__(env)
        else:
            self.env = env
        if not hasattr(env, "action_space") or not hasattr(env.action_space, "nvec"):
            raise TypeError("DiscreteActionWrapper requires a MultiDiscrete-like action space.")
        self._nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        self.observation_space = env.observation_space
        if spaces is not None:
            self.action_space = spaces.Discrete(int(np.prod(self._nvec)))
        else:
            self.action_space = type("DiscreteSpace", (), {"n": int(np.prod(self._nvec))})()

    @property
    def unwrapped(self):
        return self.env

    @property
    def state(self):
        return getattr(self.env, "state", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action: int):
        a = int(action)
        next_target_choice = a % int(self._nvec[1])
        action_type = a // int(self._nvec[1])
        return np.array([action_type, next_target_choice], dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.env, name)


class FixedOrderFreeTimeWrapper(gym.Wrapper if gym is not None else object):
    """Keeps problem order fixed while letting the model choose when to move on."""

    def __init__(self, env, min_time_per_problem_sec: float = 0.0):
        if gym is not None:
            super().__init__(env)
        else:
            self.env = env
        self.min_time_per_problem_sec = float(min_time_per_problem_sec)
        self.observation_space = env.observation_space
        if spaces is not None:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = type("DiscreteSpace", (), {"n": 2})()

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def state(self):
        return getattr(self.env, "state", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        choice = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
        current_idx = int(self.state.current_problem_idx)
        if current_idx >= self.num_problems - 1:
            return self.env.encode_solve_more_action()

        progress = self.state.progress[current_idx]
        if float(progress.time_spent_sec) + 1e-9 < self.min_time_per_problem_sec:
            return self.env.encode_solve_more_action()

        remaining_future_problems = int(self.num_problems - current_idx - 1)
        reserved_future_time = remaining_future_problems * self.min_time_per_problem_sec
        remaining_after_solve = max(
            float(self.state.remaining_time_sec) - float(self.env.action_time_unit_sec),
            0.0,
        )
        if choice == 0 and remaining_after_solve + 1e-9 >= reserved_future_time:
            return self.env.encode_solve_more_action()
        return self.env.encode_next_action(current_idx + 1)

    def __getattr__(self, name):
        return getattr(self.env, name)


class EqualTimeFreeOrderWrapper(gym.Wrapper if gym is not None else object):
    """Keeps per-problem work time fixed while letting the model choose destinations."""

    def __init__(self, env, time_budget_sec: float):
        if gym is not None:
            super().__init__(env)
        else:
            self.env = env
        self.time_budget_sec = float(time_budget_sec)
        self.observation_space = env.observation_space
        if spaces is not None:
            self.action_space = spaces.Discrete(int(env.num_problems))
        else:
            self.action_space = type("DiscreteSpace", (), {"n": int(env.num_problems)})()

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def state(self):
        return getattr(self.env, "state", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        current_idx = int(self.state.current_problem_idx)
        progress = self.state.progress[current_idx]
        if float(progress.time_spent_sec) + 1e-9 < self.time_budget_sec:
            return self.env.encode_solve_more_action()

        target_idx = int(np.asarray(action, dtype=np.int64).reshape(-1)[0]) % int(self.num_problems)
        if target_idx == current_idx:
            target_idx = (current_idx + 1) % int(self.num_problems)
        return self.env.encode_next_action(target_idx)

    def __getattr__(self, name):
        return getattr(self.env, name)


def _apply_strategy_constraint(env, config: dict[str, Any]):
    strategy_cfg = dict(config.get("training", {}).get("strategy_constraint", {}) or {})
    name = str(strategy_cfg.get("name", "") or "").lower()
    if name in {"", "none", "null"}:
        return env
    if name == "fixed_order_free_time":
        min_time = float(strategy_cfg.get("min_time_per_problem_sec", 0.0))
        return FixedOrderFreeTimeWrapper(env, min_time_per_problem_sec=min_time)
    if name == "equal_time_free_order":
        budget = float(strategy_cfg.get("time_budget_sec", env.total_time_sec / max(env.num_problems, 1)))
        return EqualTimeFreeOrderWrapper(env, time_budget_sec=budget)
    raise ValueError(
        "training.strategy_constraint.name must be one of: "
        "fixed_order_free_time, equal_time_free_order, none"
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dirs(base_dir: str) -> dict[str, str]:
    model_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    eval_dir = os.path.join(base_dir, "eval")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return {"base": base_dir, "model": model_dir, "log": log_dir, "eval": eval_dir}


def _build_env(config: dict[str, Any], for_dqn: bool = False, seed: int | None = None):
    env = ExamStrategyEnv(config=config, random_seed=seed)
    env = _apply_strategy_constraint(env, config)
    if for_dqn and hasattr(env.action_space, "nvec"):
        env = DiscreteActionWrapper(env)
    return env


def _assert_sb3_available() -> None:
    if not HAS_GYM:
        raise ImportError("gymnasium or gym is required for RL training.")
    if PPO is None or DQN is None or DummyVecEnv is None or CheckpointCallback is None or BaseCallback is None or CallbackList is None:
        raise ImportError(
            "stable-baselines3 is not installed. Install with: pip install stable-baselines3[extra]"
        )


def _select_torch_device() -> str:
    return _select_torch_device_from_value("auto")


def _select_torch_device_from_value(device_pref: str) -> str:
    pref = str(device_pref).lower()
    if torch is None:
        return "cpu"

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("Requested device 'cuda' but CUDA is not available.")
    if pref == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        raise ValueError("Requested device 'mps' but MPS is not available.")
    if pref != "auto":
        raise ValueError("training.device must be one of: auto, cpu, cuda, mps")

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


class ScoreLogCallback(BaseCallback if BaseCallback is not None else object):
    """Periodically evaluates the current model and appends a JSONL entry to log_path.

    Each record: {"timestep": N, "mean_score": X, "mean_reward": Y, "mean_coverage_fraction": Z}

    Evaluation uses deterministic policy over n_eval_episodes full episodes.
    Kept lightweight: low n_eval_episodes (default 5) to not dominate training time.
    """

    def __init__(
        self,
        config: dict[str, Any],
        log_path: str,
        eval_freq: int,
        n_eval_episodes: int = 5,
        algorithm: str = "ppo",
        seed: int = 0,
    ) -> None:
        if BaseCallback is not None:
            super().__init__()
        self.config = config
        self.log_path = log_path
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = max(int(n_eval_episodes), 1)
        self.algorithm = algorithm
        self.seed = seed
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = evaluate_trained_model(
            model=self.model,
            config=self.config,
            n_episodes=self.n_eval_episodes,
            algorithm=self.algorithm,
            seed=self.seed,
        )
        entry = {
            "timestep": self.num_timesteps,
            "mean_score": metrics["mean_score"],
            "mean_reward": metrics["mean_reward"],
            "mean_coverage_fraction": metrics["mean_coverage_fraction"],
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True


class ProgressPrinterCallback(BaseCallback if BaseCallback is not None else object):
    def __init__(self, total_timesteps: int, print_freq: int) -> None:
        if BaseCallback is not None:
            super().__init__()
        self.total_timesteps = max(int(total_timesteps), 1)
        self.print_freq = max(int(print_freq), 1)
        self._last_print = 0
        self._pbar = None
        self._last_pbar_step = 0

    def _on_training_start(self) -> None:
        if tqdm is not None:
            self._pbar = tqdm(total=self.total_timesteps, desc="train", unit="step", dynamic_ncols=True)

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            remaining = self.num_timesteps - self._last_pbar_step
            if remaining > 0:
                self._pbar.update(remaining)
            self._pbar.close()
            self._pbar = None

    def _on_step(self) -> bool:
        if self._pbar is not None:
            delta = self.num_timesteps - self._last_pbar_step
            if delta > 0:
                self._pbar.update(delta)
                self._last_pbar_step = self.num_timesteps
        if self.num_timesteps - self._last_print >= self.print_freq or self.num_timesteps >= self.total_timesteps:
            progress = min(100.0, 100.0 * self.num_timesteps / self.total_timesteps)
            if self._pbar is None:
                print(f"[train] {self.num_timesteps}/{self.total_timesteps} steps ({progress:.1f}%)")
            self._last_print = self.num_timesteps
        return True


def _build_callbacks(
    paths: dict[str, str],
    train_cfg: dict[str, Any],
    config: dict[str, Any] | None = None,
    algorithm: str = "ppo",
    base_seed: int = 42,
):
    total_steps = int(train_cfg.get("total_steps", 500000))
    checkpoint_freq = int(train_cfg.get("checkpoint_freq", max(1000, total_steps // 10)))
    progress_log_freq = int(train_cfg.get("progress_log_freq", max(1000, total_steps // 20)))

    checkpoint = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=paths["model"],
        name_prefix=str(train_cfg.get("checkpoint_prefix", "ckpt")),
    )
    progress = ProgressPrinterCallback(total_timesteps=total_steps, print_freq=progress_log_freq)
    callbacks: list = [checkpoint, progress]

    score_log_freq = int(train_cfg.get("score_log_freq", 0))
    if config is not None and score_log_freq > 0:
        n_score_eval = int(train_cfg.get("score_log_eval_episodes", 5))
        log_path = os.path.join(paths["eval"], "score_curve.jsonl")
        callbacks.append(
            ScoreLogCallback(
                config=config,
                log_path=log_path,
                eval_freq=score_log_freq,
                n_eval_episodes=n_score_eval,
                algorithm=algorithm,
                seed=base_seed,
            )
        )

    return CallbackList(callbacks)


def evaluate_trained_model(
    model,
    config: dict[str, Any],
    n_episodes: int = 30,
    algorithm: str = "ppo",
    seed: int = 42,
) -> dict[str, float]:
    is_dqn = algorithm.lower() == "dqn"
    solved_criteria = solved_criteria_from_config(config)
    totals = {
        "reward": [],
        "score": [],
        "solved_count": [],
        "visited_count": [],
        "coverage_fraction": [],
        "top1_time_share": [],
        "top2_time_share": [],
        "remaining_time_sec": [],
        "steps": [],
    }

    for ep in range(n_episodes):
        env = _build_env(config=config, for_dqn=is_dqn, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)

        state = env.state if hasattr(env, "state") else env.unwrapped.state
        assert state is not None
        problem_times = [float(p.time_spent_sec) for p in state.progress]
        total_time = float(sum(problem_times))
        sorted_times = sorted(problem_times, reverse=True)
        top1_time = sorted_times[0] if sorted_times else 0.0
        top2_time = sum(sorted_times[:2]) if sorted_times else 0.0
        totals["reward"].append(ep_reward)
        totals["score"].append(float(state.total_score))
        totals["solved_count"].append(float(state.solved_count(env.problems, **solved_criteria)))
        totals["visited_count"].append(float(state.visited_count()))
        totals["coverage_fraction"].append(float(state.coverage_fraction()))
        totals.setdefault("objective_dominance_rate", []).append(float(state.objective_dominance_rate(env.problems)))
        totals.setdefault("mean_subjective_confidence", []).append(float(state.mean_subjective_confidence(env.problems)))
        totals.setdefault("subjective_solved_rate", []).append(float(state.subjective_solved_rate(env.problems, **solved_criteria)))
        totals.setdefault("objective_solved_rate", []).append(float(state.objective_solved_rate(env.problems, **solved_criteria)))
        totals["top1_time_share"].append(float(top1_time / total_time) if total_time > 0 else 0.0)
        totals["top2_time_share"].append(float(top2_time / total_time) if total_time > 0 else 0.0)
        totals["remaining_time_sec"].append(float(state.remaining_time_sec))
        totals["steps"].append(float(state.step_count))

    return {
        "episodes": float(n_episodes),
        "mean_reward": float(np.mean(totals["reward"])),
        "mean_score": float(np.mean(totals["score"])),
        "mean_solved_count": float(np.mean(totals["solved_count"])),
        "mean_visited_count": float(np.mean(totals["visited_count"])),
        "mean_coverage_fraction": float(np.mean(totals["coverage_fraction"])),
        "mean_objective_dominance_rate": float(np.mean(totals["objective_dominance_rate"])),
        "mean_subjective_confidence": float(np.mean(totals["mean_subjective_confidence"])),
        "mean_subjective_solved_rate": float(np.mean(totals["subjective_solved_rate"])),
        "mean_objective_solved_rate": float(np.mean(totals["objective_solved_rate"])),
        "mean_top1_time_share": float(np.mean(totals["top1_time_share"])),
        "mean_top2_time_share": float(np.mean(totals["top2_time_share"])),
        "mean_remaining_time_sec": float(np.mean(totals["remaining_time_sec"])),
        "mean_steps": float(np.mean(totals["steps"])),
    }


def train_ppo(config: dict[str, Any], output_root: str = "runs", run_name: str | None = None):
    _assert_sb3_available()
    run_name = run_name or f"ppo_{_timestamp()}"
    paths = _ensure_dirs(os.path.join(output_root, run_name))

    base_seed = int(config.get("experiment", {}).get("seed", 42))
    vec_env = DummyVecEnv([lambda: _build_env(config=config, for_dqn=False, seed=base_seed)])
    ppo_cfg = config.get("ppo", {})
    train_cfg = config.get("training", {})
    device = _select_torch_device_from_value(str(train_cfg.get("device", "auto")))
    total_steps = int(train_cfg.get("total_steps", 500000))
    print(f"[train_ppo] device={device} total_steps={total_steps} eval_episodes={int(train_cfg.get('eval_episodes', 100))}")
    print(f"[train_ppo] output_dir={paths['base']}")

    net_arch = ppo_cfg.get("net_arch", [64, 64])
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        policy_kwargs=dict(net_arch=list(net_arch)),
        device=device,
        seed=base_seed,
        tensorboard_log=paths["log"],
        verbose=0,
    )

    callbacks = _build_callbacks(
        paths=paths,
        train_cfg={**train_cfg, "checkpoint_prefix": "ppo_ckpt"},
        config=config,
        algorithm="ppo",
        base_seed=base_seed,
    )
    model.learn(total_timesteps=total_steps, callback=callbacks)

    final_model_path = os.path.join(paths["model"], "ppo_final")
    model.save(final_model_path)
    print(f"[train_ppo] final_model={final_model_path}.zip")

    metrics = evaluate_trained_model(
        model=model,
        config=config,
        n_episodes=int(train_cfg.get("eval_episodes", 100)),
        algorithm="ppo",
        seed=base_seed,
    )

    with open(os.path.join(paths["base"], "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(paths["eval"], "ppo_eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_ppo] eval_mean_score={metrics['mean_score']:.4f} eval_mean_reward={metrics['mean_reward']:.4f}")

    return {"paths": paths, "final_model_path": final_model_path, "eval": metrics}


def train_dqn(config: dict[str, Any], output_root: str = "runs", run_name: str | None = None):
    _assert_sb3_available()
    run_name = run_name or f"dqn_{_timestamp()}"
    paths = _ensure_dirs(os.path.join(output_root, run_name))

    base_seed = int(config.get("experiment", {}).get("seed", 42))
    vec_env = DummyVecEnv([lambda: _build_env(config=config, for_dqn=True, seed=base_seed)])
    dqn_cfg = config.get("dqn", {})
    train_cfg = config.get("training", {})
    device = _select_torch_device_from_value(str(train_cfg.get("device", "auto")))
    total_steps = int(train_cfg.get("total_steps", 500000))
    print(f"[train_dqn] device={device} total_steps={total_steps} eval_episodes={int(train_cfg.get('eval_episodes', 100))}")
    print(f"[train_dqn] output_dir={paths['base']}")

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(dqn_cfg.get("learning_rate", 1e-4)),
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        buffer_size=int(dqn_cfg.get("buffer_size", 100000)),
        learning_starts=int(dqn_cfg.get("learning_starts", 10000)),
        batch_size=int(dqn_cfg.get("batch_size", 128)),
        train_freq=int(dqn_cfg.get("train_freq", 4)),
        target_update_interval=int(dqn_cfg.get("target_update_interval", 1000)),
        exploration_initial_eps=float(dqn_cfg.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(dqn_cfg.get("exploration_final_eps", 0.05)),
        exploration_fraction=float(dqn_cfg.get("exploration_fraction", 0.2)),
        device=device,
        seed=base_seed,
        tensorboard_log=paths["log"],
        verbose=0,
    )

    callbacks = _build_callbacks(
        paths=paths,
        train_cfg={**train_cfg, "checkpoint_prefix": "dqn_ckpt"},
        config=config,
        algorithm="dqn",
        base_seed=base_seed,
    )
    model.learn(total_timesteps=total_steps, callback=callbacks)

    final_model_path = os.path.join(paths["model"], "dqn_final")
    model.save(final_model_path)
    print(f"[train_dqn] final_model={final_model_path}.zip")

    metrics = evaluate_trained_model(
        model=model,
        config=config,
        n_episodes=int(train_cfg.get("eval_episodes", 100)),
        algorithm="dqn",
        seed=base_seed,
    )

    with open(os.path.join(paths["base"], "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(paths["eval"], "dqn_eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_dqn] eval_mean_score={metrics['mean_score']:.4f} eval_mean_reward={metrics['mean_reward']:.4f}")

    return {"paths": paths, "final_model_path": final_model_path, "eval": metrics}


def train_from_config(
    config: dict[str, Any],
    output_root: str = "runs",
    run_name: str | None = None,
):
    algo = str(config.get("training", {}).get("algorithm", "ppo")).lower()
    if algo == "ppo":
        return train_ppo(config, output_root=output_root, run_name=run_name)
    if algo == "dqn":
        return train_dqn(config, output_root=output_root, run_name=run_name)
    raise ValueError("training.algorithm must be either 'ppo' or 'dqn'.")


def train_multi_seed(
    config: dict[str, Any],
    seeds: list[int] | tuple[int, ...] = (42, 123, 2024),
    output_root: str = "runs",
    run_prefix: str | None = None,
) -> dict[str, Any]:
    """Train the same algorithm with multiple seeds and aggregate results.

    For each seed:
      - Overrides config['experiment']['seed'] with the given seed.
      - Saves the run to runs/<run_prefix>_seed<N>_<timestamp>/.

    After all runs, writes a multiseed_summary.json to
    runs/<run_prefix>_multiseed_<timestamp>/ with mean±std across seeds.

    Returns the summary dict.
    """
    algo = str(config.get("training", {}).get("algorithm", "ppo")).lower()
    run_prefix = run_prefix or algo
    timestamp = _timestamp()

    all_results: list[dict[str, Any]] = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config.setdefault("experiment", {})["seed"] = int(seed)
        run_name = f"{run_prefix}_seed{seed}_{timestamp}"
        print(f"\n[train_multi_seed] === seed={seed}  run={run_name} ===")
        result = train_from_config(seed_config, output_root=output_root, run_name=run_name)
        result["seed"] = int(seed)
        all_results.append(result)

    score_list = [r["eval"]["mean_score"] for r in all_results]
    reward_list = [r["eval"]["mean_reward"] for r in all_results]
    coverage_list = [r["eval"]["mean_coverage_fraction"] for r in all_results]

    summary: dict[str, Any] = {
        "algorithm": algo,
        "seeds": [r["seed"] for r in all_results],
        "mean_score": float(np.mean(score_list)),
        "std_score": float(np.std(score_list)),
        "mean_reward": float(np.mean(reward_list)),
        "std_reward": float(np.std(reward_list)),
        "mean_coverage_fraction": float(np.mean(coverage_list)),
        "per_seed": [
            {
                "seed": r["seed"],
                "mean_score": r["eval"]["mean_score"],
                "mean_reward": r["eval"]["mean_reward"],
                "mean_coverage_fraction": r["eval"]["mean_coverage_fraction"],
                "final_model_path": r["final_model_path"],
                "run_dir": r["paths"]["base"],
            }
            for r in all_results
        ],
    }

    summary_dir = os.path.join(output_root, f"{run_prefix}_multiseed_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "multiseed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[train_multi_seed] summary → {summary_path}")
    print(f"[train_multi_seed] mean_score={summary['mean_score']:.4f} ± {summary['std_score']:.4f}")
    print(f"[train_multi_seed] per-seed scores: {[f'{s:.4f}' for s in score_list]}")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Train RL model for exam strategy optimization.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="runs")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed training, e.g. '42,123,2024'. "
             "If omitted, runs a single training using config's experiment.seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        result = train_multi_seed(cfg, seeds=seeds, output_root=args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))
    else:
        result = train_from_config(cfg, output_root=args.output)
        print(json.dumps(result["eval"], indent=2))
