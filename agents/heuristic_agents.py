from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from env.dynamics import confidence_curve, confidence_static_params, expected_total_score
from env.exam_env import ExamStrategyEnv
from env.problem import Problem
from env.state import ProblemStatus, solved_criteria_from_config
from env.time_allocation_env import TimeAllocationEnv


HeuristicFn = Callable[[ExamStrategyEnv], np.ndarray]


@dataclass
class EpisodeStats:
    total_reward: float
    total_score: float
    solved_count: int
    visited_count: int
    coverage_fraction: float
    objective_dominance_rate: float
    mean_subjective_confidence: float
    subjective_solved_rate: float
    objective_solved_rate: float
    top1_time_share: float
    top2_time_share: float
    remaining_time_sec: float
    steps: int
    done: bool


# ---------------------------------------------------------------------------
# Difficulty anchor helper (consistent with Phase-2 dynamics)
# ---------------------------------------------------------------------------

def _difficulty_anchor(problem: Problem) -> float:
    """Return the difficulty anchor used for time budgets.

    Mirrors env/dynamics._difficulty_anchor: prefers 1 - correct_rate
    when available, falls back to problem.difficulty.
    """
    if problem.correct_rate is not None:
        return max(0.0, min(1.0, 1.0 - problem.correct_rate))
    return max(0.0, min(1.0, problem.difficulty))


def target_time_budget(problem: Problem, policy_name: str) -> float:
    hardness = _difficulty_anchor(problem)
    base_budget = 45.0 + 90.0 * hardness
    if policy_name == "equal_time":
        return base_budget
    if policy_name == "index_order":
        return base_budget
    if policy_name == "easy_first":
        return base_budget * (0.90 if problem.score <= 3 else 0.75)
    if policy_name == "high_score_first":
        return base_budget * (1.20 if problem.score >= 4 else 0.85)
    if policy_name == "score_time_ratio":
        return base_budget * (1.05 if problem.score >= 4 else 0.80)
    return base_budget


# ---------------------------------------------------------------------------
# Marginal gain helper
# ---------------------------------------------------------------------------

def _marginal_gain_per_second(env: ExamStrategyEnv, problem_idx: int) -> float:
    """Expected score gain per second from spending one action unit on problem_idx.

    Formula:
        gain(same problem)   = score * delta_conf / action_time
        gain(switch + solve) = score * delta_conf / (action_time + switch_cost)

    delta_conf = max(0, conf_after - conf_now)
    conf_after = confidence_curve(t_now + action_time_unit)

    Monotonicity of confidence_curve means delta_conf >= 0 always holds,
    but we guard with max(0, ...) for safety.
    """
    assert env.state is not None and env.current_student is not None
    problem = env.problems[problem_idx]
    progress = env.state.progress[problem_idx]
    t_now = float(progress.time_spent_sec)
    dt = env.action_time_unit_sec

    # Switch cost only when moving to a different problem
    current_idx = env.state.current_problem_idx
    switch_cost = env.switch_time_sec if problem_idx != current_idx else 0.0
    total_cost = dt + switch_cost
    if total_cost <= 0.0:
        return 0.0

    conf_now = float(progress.effective_confidence(problem))
    conf_after = float(
        confidence_curve(
            problem=problem,
            student=env.current_student,
            time_spent=t_now + dt,
            dynamics_cfg=env.dynamics_cfg,
        )
    )
    # Respect the monotonicity enforced by solve_more
    conf_after = max(conf_now, conf_after)
    delta_conf = conf_after - conf_now

    return float(problem.score) * delta_conf / total_cost


# ---------------------------------------------------------------------------
# Start-problem selection (for allow_agent_selected_start_problem mode)
# ---------------------------------------------------------------------------

def _select_start_problem(env: ExamStrategyEnv, policy_name: str) -> int:
    """Select the initial problem when no problem has been started yet.

    All problems are candidates. Uses the same priority logic as
    _select_next_problem but without a "current" exclusion.
    """
    candidates = list(range(env.num_problems))
    if not candidates:
        return 0

    if policy_name == "easy_first":
        return min(candidates, key=lambda idx: (_difficulty_anchor(env.problems[idx]), env.problems[idx].score, idx))

    if policy_name == "high_score_first":
        return max(candidates, key=lambda idx: (env.problems[idx].score, -_difficulty_anchor(env.problems[idx]), -idx))

    if policy_name in {"score_time_ratio", "score_time_ratio_greedy"}:
        return max(
            candidates,
            key=lambda idx: env.problems[idx].score / max(_difficulty_anchor(env.problems[idx]), 0.05),
        )

    if policy_name == "marginal_gain_greedy":
        # At t=0, all problems have the same time spent (0). Pick highest gain.
        return max(candidates, key=lambda idx: _marginal_gain_per_second(env, idx))

    # equal_time / index_order / fallback: start at problem 0
    return 0


# ---------------------------------------------------------------------------
# Core action selection
# ---------------------------------------------------------------------------

def _current_budget_action(env: ExamStrategyEnv, policy_name: str) -> np.ndarray:
    assert env.state is not None

    # ── Not-started state (allow_agent_selected_start_problem=true) ──────────
    if env.state.current_problem_idx == -1:
        first_target = _select_start_problem(env, policy_name)
        return env.encode_select_start_action(first_target)
    # ─────────────────────────────────────────────────────────────────────────

    current_idx = env.state.current_problem_idx
    problem = env.problems[current_idx]
    if policy_name == "equal_time":
        reserved_switch_time = env.switch_time_sec * max(env.num_problems - 1, 0)
        budget = max((env.total_time_sec - reserved_switch_time) / max(env.num_problems, 1), 0.0)
    else:
        budget = target_time_budget(problem, policy_name)
    progress = env.state.progress[current_idx]
    if progress.time_spent_sec < budget:
        return env.encode_solve_more_action()
    return env.encode_next_action(_select_next_problem(env, policy_name))


def _marginal_gain_action(env: ExamStrategyEnv) -> np.ndarray:
    """Greedy 1-step lookahead: pick the action with highest expected gain/sec.

    At each step, we evaluate:
    - solve_more on current problem
    - switch to each other problem and spend one action unit there

    Returns solve_more if current has the highest marginal gain, otherwise
    switches to the problem with the best gain.
    """
    assert env.state is not None

    # ── Not-started state ───────────────────────────────────────────────────
    if env.state.current_problem_idx == -1:
        first_target = _select_start_problem(env, "marginal_gain_greedy")
        return env.encode_select_start_action(first_target)
    # ────────────────────────────────────────────────────────────────────────

    current_idx = env.state.current_problem_idx
    best_idx = current_idx
    best_gain = _marginal_gain_per_second(env, current_idx)

    for idx in range(env.num_problems):
        if idx == current_idx:
            continue
        gain = _marginal_gain_per_second(env, idx)
        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    if best_idx == current_idx:
        return env.encode_solve_more_action()
    return env.encode_next_action(best_idx)


def _select_next_problem(env: ExamStrategyEnv, policy_name: str) -> int:
    assert env.state is not None
    candidates = [
        idx
        for idx, progress in enumerate(env.state.progress)
        if idx != env.state.current_problem_idx
    ]
    if not candidates:
        return env.state.current_problem_idx
    if policy_name == "index_order":
        return min(candidates)
    if policy_name == "equal_time":
        unvisited = [idx for idx in candidates if env.state.progress[idx].status == ProblemStatus.NOT_VISITED]
        if unvisited:
            return min(unvisited)
        return min(candidates)
    if policy_name == "easy_first":
        return min(candidates, key=lambda idx: (_difficulty_anchor(env.problems[idx]), env.problems[idx].score, idx))
    if policy_name == "high_score_first":
        return max(candidates, key=lambda idx: (env.problems[idx].score, -_difficulty_anchor(env.problems[idx]), -idx))
    if policy_name == "score_time_ratio":
        return max(
            candidates,
            key=lambda idx: env.problems[idx].score / max(_difficulty_anchor(env.problems[idx]), 0.05),
        )
    return min(candidates)


# ---------------------------------------------------------------------------
# Public policy functions
# ---------------------------------------------------------------------------

def policy_index_order(env: ExamStrategyEnv) -> np.ndarray:
    return _current_budget_action(env, "index_order")


def policy_equal_time(env: ExamStrategyEnv) -> np.ndarray:
    return _current_budget_action(env, "equal_time")


def policy_easy_first(env: ExamStrategyEnv) -> np.ndarray:
    return _current_budget_action(env, "easy_first")


def policy_high_score_first(env: ExamStrategyEnv) -> np.ndarray:
    return _current_budget_action(env, "high_score_first")


def policy_expected_score_time_ratio(env: ExamStrategyEnv) -> np.ndarray:
    return _current_budget_action(env, "score_time_ratio")


def policy_marginal_gain_greedy(env: ExamStrategyEnv) -> np.ndarray:
    """At every step, pick the action with the highest expected score gain per second.

    Evaluates:
    - solve_more on the current problem
    - switch to each other problem (paying switch_time_sec) and spend one action unit

    The policy naturally balances revisiting saturated problems vs exploring new ones
    because marginal gain falls as confidence approaches the ceiling.
    """
    return _marginal_gain_action(env)


def policy_random(env: ExamStrategyEnv) -> np.ndarray:
    """Uniformly random baseline: each step randomly picks solve_more or a random other problem.

    Uses the environment's own RNG for reproducibility.
    """
    assert env.state is not None

    # not-started mode: pick a random start problem
    if env.state.current_problem_idx == -1:
        target = int(env.rng.integers(0, env.num_problems))
        return env.encode_select_start_action(target)

    # 50/50 between solve_more and move
    if int(env.rng.integers(0, 2)) == 0:
        return env.encode_solve_more_action()

    others = [i for i in range(env.num_problems) if i != env.state.current_problem_idx]
    target = int(env.rng.choice(others))
    return env.encode_next_action(target)


HEURISTIC_POLICIES: dict[str, HeuristicFn] = {
    "random": policy_random,
    "equal_time": policy_equal_time,
    "index_order": policy_index_order,
    "easy_first": policy_easy_first,
    "high_score_first": policy_high_score_first,
    "score_time_ratio": policy_expected_score_time_ratio,
    "marginal_gain_greedy": policy_marginal_gain_greedy,
}


def heuristic_action(env: ExamStrategyEnv, policy_name: str) -> np.ndarray:
    selector = HEURISTIC_POLICIES.get(policy_name)
    if selector is None:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")
    return selector(env)


def run_heuristic_episode(
    env: ExamStrategyEnv,
    policy_name: str,
    max_steps: int = 10000,
    reset_seed: int | None = None,
    solved_criteria: dict[str, float] | None = None,
) -> EpisodeStats:
    _, _ = env.reset(seed=reset_seed)
    solved_criteria = solved_criteria or {}

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = heuristic_action(env, policy_name)
        _, r, terminated, truncated, _ = env.step(action)
        total_reward += float(r)
        done = bool(terminated or truncated)
        steps += 1

    assert env.state is not None
    problem_times = [float(p.time_spent_sec) for p in env.state.progress]
    total_time = float(sum(problem_times))
    sorted_times = sorted(problem_times, reverse=True)
    top1_time = sorted_times[0] if sorted_times else 0.0
    top2_time = sum(sorted_times[:2]) if sorted_times else 0.0
    return EpisodeStats(
        total_reward=float(total_reward),
        total_score=float(env.state.total_score),
        solved_count=int(env.state.solved_count(env.problems, **solved_criteria)),
        visited_count=int(env.state.visited_count()),
        coverage_fraction=float(env.state.coverage_fraction()),
        objective_dominance_rate=float(env.state.objective_dominance_rate(env.problems)),
        mean_subjective_confidence=float(env.state.mean_subjective_confidence(env.problems)),
        subjective_solved_rate=float(env.state.subjective_solved_rate(env.problems, **solved_criteria)),
        objective_solved_rate=float(env.state.objective_solved_rate(env.problems, **solved_criteria)),
        top1_time_share=float(top1_time / total_time) if total_time > 0 else 0.0,
        top2_time_share=float(top2_time / total_time) if total_time > 0 else 0.0,
        remaining_time_sec=float(env.state.remaining_time_sec),
        steps=int(steps),
        done=bool(done),
    )


def evaluate_heuristic_policy(
    env_factory: Callable[[], ExamStrategyEnv],
    policy_name: str,
    episodes: int = 50,
    seed: int = 42,
) -> dict:
    solved_criteria = solved_criteria_from_config(getattr(env_factory(), "config", None))
    metrics = {
        "reward": [],
        "score": [],
        "solved_count": [],
        "visited_count": [],
        "coverage_fraction": [],
        "objective_dominance_rate": [],
        "mean_subjective_confidence": [],
        "subjective_solved_rate": [],
        "objective_solved_rate": [],
        "top1_time_share": [],
        "top2_time_share": [],
        "remaining_time_sec": [],
        "steps": [],
    }
    for ep in range(episodes):
        env = env_factory()
        stats = run_heuristic_episode(
            env=env,
            policy_name=policy_name,
            reset_seed=seed + ep,
            solved_criteria=solved_criteria,
        )
        metrics["reward"].append(stats.total_reward)
        metrics["score"].append(stats.total_score)
        metrics["solved_count"].append(stats.solved_count)
        metrics["visited_count"].append(stats.visited_count)
        metrics["coverage_fraction"].append(stats.coverage_fraction)
        metrics["objective_dominance_rate"].append(stats.objective_dominance_rate)
        metrics["mean_subjective_confidence"].append(stats.mean_subjective_confidence)
        metrics["subjective_solved_rate"].append(stats.subjective_solved_rate)
        metrics["objective_solved_rate"].append(stats.objective_solved_rate)
        metrics["top1_time_share"].append(stats.top1_time_share)
        metrics["top2_time_share"].append(stats.top2_time_share)
        metrics["remaining_time_sec"].append(stats.remaining_time_sec)
        metrics["steps"].append(stats.steps)

    return {
        "policy": policy_name,
        "episodes": episodes,
        "mean_reward": float(np.mean(metrics["reward"])),
        "mean_score": float(np.mean(metrics["score"])),
        "mean_solved_count": float(np.mean(metrics["solved_count"])),
        "mean_visited_count": float(np.mean(metrics["visited_count"])),
        "mean_coverage_fraction": float(np.mean(metrics["coverage_fraction"])),
        "mean_objective_dominance_rate": float(np.mean(metrics["objective_dominance_rate"])),
        "mean_subjective_confidence": float(np.mean(metrics["mean_subjective_confidence"])),
        "mean_subjective_solved_rate": float(np.mean(metrics["subjective_solved_rate"])),
        "mean_objective_solved_rate": float(np.mean(metrics["objective_solved_rate"])),
        "mean_top1_time_share": float(np.mean(metrics["top1_time_share"])),
        "mean_top2_time_share": float(np.mean(metrics["top2_time_share"])),
        "mean_remaining_time_sec": float(np.mean(metrics["remaining_time_sec"])),
        "mean_steps": float(np.mean(metrics["steps"])),
    }


def evaluate_all_heuristics(
    env_factory: Callable[[], ExamStrategyEnv],
    episodes: int = 50,
    seed: int = 42,
) -> list[dict]:
    return [
        evaluate_heuristic_policy(env_factory=env_factory, policy_name=name, episodes=episodes, seed=seed)
        for name in HEURISTIC_POLICIES
    ]


# ===========================================================================
# Time-allocation baselines
# ===========================================================================

AllocationPolicyFn = Callable[[TimeAllocationEnv], int]

ALLOCATION_POLICIES = ["equal_time", "difficulty_prior", "greedy_marginal_gain"]


def _allocation_marginal_gain(env: TimeAllocationEnv, problem_idx: int) -> float:
    """Expected score gain from one more action unit on problem_idx."""
    if env.state is None or env.current_student is None or env._mg_params_cache is None:
        return 0.0
    progress = env.state.progress[problem_idx]
    problem = env.problems[problem_idx]
    mg_floor, mg_static_logit, mg_alpha, mg_tau, mg_score_norm = env._mg_params_cache[problem_idx]
    import math
    t_future = float(progress.time_spent_sec) + env.action_time_unit_sec
    logit_f = mg_static_logit + mg_alpha * math.log(1.0 + t_future / max(mg_tau, 1.0))
    sig_f = (1.0 / (1.0 + math.exp(-logit_f)) if logit_f >= 0
             else math.exp(logit_f) / (1.0 + math.exp(logit_f)))
    p_future = max(mg_floor, min(1.0, mg_floor + (1.0 - mg_floor) * sig_f))
    conf_now = float(progress.effective_confidence(problem))
    return float(mg_score_norm * max(p_future - conf_now, 0.0))


def allocation_policy_equal_time(env: TimeAllocationEnv) -> int:
    """Round-robin: allocate to whichever problem has the least time so far."""
    if env.state is None:
        return 0
    times = [float(p.time_spent_sec) for p in env.state.progress]
    return int(np.argmin(times))


def allocation_policy_difficulty_prior(env: TimeAllocationEnv) -> int:
    """Allocate proportional to difficulty: harder problems get more time.

    Implementation: always allocate to the problem whose current time share
    is furthest below its target share.
    """
    if env.state is None:
        return 0
    difficulty_map = {"하": 0.1, "중하": 0.2, "중": 0.4, "중상": 0.6, "상": 0.8, "최상": 1.0}
    weights = np.array(
        [difficulty_map.get(p.difficulty_level, 0.5) for p in env.problems], dtype=float
    )
    weights = weights / max(weights.sum(), 1e-9)
    times = np.array([float(p.time_spent_sec) for p in env.state.progress], dtype=float)
    total = times.sum()
    if total <= 0:
        return int(np.argmax(weights))
    current_shares = times / total
    deficits = weights - current_shares
    return int(np.argmax(deficits))


def allocation_policy_greedy_marginal_gain(env: TimeAllocationEnv) -> int:
    """Greedy: allocate to the problem with highest current marginal gain.

    This is optimal for the separable concave time-allocation problem
    (discrete water-filling).
    """
    if env.state is None:
        return 0
    gains = [_allocation_marginal_gain(env, i) for i in range(env.num_problems)]
    return int(np.argmax(gains))


_ALLOCATION_POLICY_FNS: dict[str, AllocationPolicyFn] = {
    "equal_time": allocation_policy_equal_time,
    "difficulty_prior": allocation_policy_difficulty_prior,
    "greedy_marginal_gain": allocation_policy_greedy_marginal_gain,
}


def evaluate_allocation_policy(
    env_factory: Callable[[], TimeAllocationEnv],
    policy_name: str,
    episodes: int = 50,
    seed: int = 42,
) -> dict:
    """Run an allocation heuristic and return summary statistics."""
    policy_fn = _ALLOCATION_POLICY_FNS[policy_name]
    solved_criteria: dict = {}

    metrics: dict[str, list] = {
        k: [] for k in [
            "reward", "score", "solved_count", "visited_count", "coverage_fraction",
            "objective_dominance_rate", "mean_subjective_confidence",
            "subjective_solved_rate", "objective_solved_rate",
            "top1_time_share", "top2_time_share", "remaining_time_sec", "steps",
        ]
    }

    for ep in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=seed + ep)
        if not solved_criteria:
            solved_criteria = solved_criteria_from_config(env.config)
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action = policy_fn(env)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)

        state = env.state
        assert state is not None
        times = [float(p.time_spent_sec) for p in state.progress]
        total_t = sum(times)
        sorted_t = sorted(times, reverse=True)
        top1 = sorted_t[0] if sorted_t else 0.0
        top2 = sum(sorted_t[:2]) if sorted_t else 0.0

        metrics["reward"].append(ep_reward)
        metrics["score"].append(float(state.total_score))
        metrics["solved_count"].append(float(state.solved_count(env.problems, **solved_criteria)))
        metrics["visited_count"].append(float(state.visited_count()))
        metrics["coverage_fraction"].append(float(state.coverage_fraction()))
        metrics["objective_dominance_rate"].append(float(state.objective_dominance_rate(env.problems)))
        metrics["mean_subjective_confidence"].append(float(state.mean_subjective_confidence(env.problems)))
        metrics["subjective_solved_rate"].append(float(state.subjective_solved_rate(env.problems, **solved_criteria)))
        metrics["objective_solved_rate"].append(float(state.objective_solved_rate(env.problems, **solved_criteria)))
        metrics["top1_time_share"].append(float(top1 / total_t) if total_t > 0 else 0.0)
        metrics["top2_time_share"].append(float(top2 / total_t) if total_t > 0 else 0.0)
        metrics["remaining_time_sec"].append(float(state.remaining_time_sec))
        metrics["steps"].append(float(state.step_count))

    return {
        "policy": policy_name,
        "episodes": episodes,
        **{f"mean_{k}": float(np.mean(v)) for k, v in metrics.items()},
    }


def evaluate_all_allocation_policies(
    env_factory: Callable[[], TimeAllocationEnv],
    episodes: int = 50,
    seed: int = 42,
) -> list[dict]:
    return [
        evaluate_allocation_policy(env_factory=env_factory, policy_name=name,
                                   episodes=episodes, seed=seed)
        for name in ALLOCATION_POLICIES
    ]
