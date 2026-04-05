from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from env.exam_env import ExamStrategyEnv
from utils.io import load_config

HAS_GYM = True
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        HAS_GYM = False
        gym = None
        spaces = None

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover
    PPO = None
    DQN = None
    CheckpointCallback = None
    DummyVecEnv = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class DiscreteActionWrapper:
    """Converts MultiDiscrete([N, 4]) -> Discrete(N*4) for DQN."""

    def __init__(self, env):
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
        action_type = a % int(self._nvec[1])
        problem_idx = a // int(self._nvec[1])
        return np.array([problem_idx, action_type], dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self.env, name)


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
    if for_dqn:
        env = DiscreteActionWrapper(env)
    return env


def _assert_sb3_available() -> None:
    if not HAS_GYM:
        raise ImportError("gymnasium or gym is required for RL training.")
    if PPO is None or DQN is None or DummyVecEnv is None or CheckpointCallback is None:
        raise ImportError(
            "stable-baselines3 is not installed. Install with: pip install stable-baselines3[extra]"
        )


def _select_torch_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def evaluate_trained_model(
    model,
    config: dict[str, Any],
    n_episodes: int = 30,
    algorithm: str = "ppo",
    seed: int = 42,
) -> dict[str, float]:
    is_dqn = algorithm.lower() == "dqn"
    totals = {"reward": [], "score": [], "solved_count": [], "remaining_time_sec": []}

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
        totals["reward"].append(ep_reward)
        totals["score"].append(float(state.total_score))
        totals["solved_count"].append(float(state.solved_count()))
        totals["remaining_time_sec"].append(float(state.remaining_time_sec))

    return {
        "episodes": float(n_episodes),
        "mean_reward": float(np.mean(totals["reward"])),
        "mean_score": float(np.mean(totals["score"])),
        "mean_solved_count": float(np.mean(totals["solved_count"])),
        "mean_remaining_time_sec": float(np.mean(totals["remaining_time_sec"])),
    }


def train_ppo(config: dict[str, Any], output_root: str = "runs", run_name: str | None = None):
    _assert_sb3_available()
    run_name = run_name or f"ppo_{_timestamp()}"
    paths = _ensure_dirs(os.path.join(output_root, run_name))

    base_seed = int(config.get("experiment", {}).get("seed", 42))
    vec_env = DummyVecEnv([lambda: _build_env(config=config, for_dqn=False, seed=base_seed)])
    ppo_cfg = config.get("ppo", {})
    train_cfg = config.get("training", {})
    device = _select_torch_device()
    print(f"[train_ppo] selected device: {device}")

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
        device=device,
        seed=base_seed,
        tensorboard_log=paths["log"],
        verbose=1,
    )

    callback = CheckpointCallback(
        save_freq=max(1000, int(train_cfg.get("total_steps", 500000)) // 10),
        save_path=paths["model"],
        name_prefix="ppo_ckpt",
    )
    model.learn(total_timesteps=int(train_cfg.get("total_steps", 500000)), callback=callback)

    final_model_path = os.path.join(paths["model"], "ppo_final")
    model.save(final_model_path)

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

    return {"paths": paths, "final_model_path": final_model_path, "eval": metrics}


def train_dqn(config: dict[str, Any], output_root: str = "runs", run_name: str | None = None):
    _assert_sb3_available()
    run_name = run_name or f"dqn_{_timestamp()}"
    paths = _ensure_dirs(os.path.join(output_root, run_name))

    base_seed = int(config.get("experiment", {}).get("seed", 42))
    vec_env = DummyVecEnv([lambda: _build_env(config=config, for_dqn=True, seed=base_seed)])
    dqn_cfg = config.get("dqn", {})
    train_cfg = config.get("training", {})
    device = _select_torch_device()
    print(f"[train_dqn] selected device: {device}")

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
        verbose=1,
    )

    callback = CheckpointCallback(
        save_freq=max(1000, int(train_cfg.get("total_steps", 500000)) // 10),
        save_path=paths["model"],
        name_prefix="dqn_ckpt",
    )
    model.learn(total_timesteps=int(train_cfg.get("total_steps", 500000)), callback=callback)

    final_model_path = os.path.join(paths["model"], "dqn_final")
    model.save(final_model_path)

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

    return {"paths": paths, "final_model_path": final_model_path, "eval": metrics}


def train_from_config(config: dict[str, Any], output_root: str = "runs"):
    algo = str(config.get("training", {}).get("algorithm", "ppo")).lower()
    if algo == "ppo":
        return train_ppo(config, output_root=output_root)
    if algo == "dqn":
        return train_dqn(config, output_root=output_root)
    raise ValueError("training.algorithm must be either 'ppo' or 'dqn'.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train RL model for exam strategy optimization.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    result = train_from_config(cfg, output_root=args.output)
    print(json.dumps(result["eval"], indent=2))
