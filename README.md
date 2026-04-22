# Optimal Time Allocation in Time-Limited Tests via Reinforcement Learning

<p align="center">
  <p align="center">
    <a href="https://github.com/JinWoong-Jung" target='_blank'>Jinwoong Jung</a>&emsp;
    <a>Yuji Lim</a>&emsp;
    <a>Sangyun Lee</a>&emsp;
    <a>Jungwoo Choi</a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University</i><br>
  </p>
   <p align="center">
    <i>2026 Introduction to Reinforcement Learning(AAI2024_01) Final Project</i><br>
  </p>
</p>


## ✨ Project Overview

This project studies how to allocate time across problems in a time-limited exam using reinforcement learning.
Instead of generating answers directly, the agent learns a policy that distributes a fixed exam-time budget over the problems.

At each step, the current environment gives the agent one time token, such as 30 seconds, and the agent chooses which problem receives that token. After time is allocated, the problem's confidence is updated by the confidence dynamics model, and the expected score changes accordingly.

The current experiments use:

- `PPO` ([Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)) and `DQN` ([Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)) as the RL algorithms
- `TimeAllocationEnv` as the time-allocation environment
- `low`, `mid`, and `high` student ability profiles, represented by a single ability parameter `theta`

The goal is not to learn the order in which problems are solved. The current setup intentionally removes ordering and revisit decisions so that the learned policy can be interpreted directly as "how much time should be invested in each problem?"

## 🛠️ Installation

### Clone

```bash
git clone https://github.com/JinWoong-Jung/Reinforcement_Learning_Final_Project_26-1.git
cd RLProject
```

### Conda

```bash
conda create -n rlproject python=3.11 -y
conda activate rlproject
pip install -r requirements.txt
```

### Main Dependencies

- Python 3.11
- `stable-baselines3`
- `gymnasium`
- `numpy`
- `torch`
- `matplotlib`

## 🧭 Problem Formulation

We formulate test-taking as a time-allocation problem.
Given a fixed exam-time budget, the agent repeatedly assigns the next time token to one of the problems.

- State: remaining time and per-problem information
- Action: select one problem to receive the next time token
- Transition: the selected problem receives additional time, and its confidence increases according to the confidence dynamics
- Objective: maximize expected total score within the time limit

The confidence score for each problem is modeled as

$$
p_i(t)=c_i+(1-c_i)\,\sigma\left(\theta-\beta d_i-\gamma a_i+\alpha \log\left(1+\frac{t}{\tau}\right)\right)
$$

where the main dynamics parameters are:

| Symbol | Meaning | Current default / definition |
| --- | --- | --- |
| $\theta$ | student ability | loaded directly from student profile |
| $d_i$ | difficulty of problem $i$ | `difficulty` field in the dataset |
| $a_i$ | ambiguity of problem $i$ | entropy of `choice_rate` |
| $c_i$ | minimum probability floor | `0.2` for objective, `0.0` for subjective |
| $\alpha$ | time gain coefficient | configured by `dynamics.alpha` |
| $\beta$ | difficulty penalty coefficient | configured by `dynamics.beta` |
| $\gamma$ | ambiguity penalty coefficient | configured by `dynamics.ambiguity_weight` |
| $\tau$ | time scale | configured by `dynamics.tau` |

For the final PPO/DQN experiment configs, the reward at each step is the change in expected utility, with a terminal bonus proportional to the final expected score:

$$
\begin{aligned}
r_t &= U(s_{t+1}) - U(s_t) \\
&\quad + \mathbf{1}_{\mathrm{terminal}} \cdot \lambda_{\mathrm{score}} \cdot \frac{U(s_T)}{\sum_i \mathrm{score}_i}
\end{aligned}
$$

where the expected utility is defined as

$$
U(s)=\sum_i \mathrm{score}_i \cdot \mathrm{confidence}_i(s)
$$

Here, $\lambda_{\mathrm{score}}$ corresponds to `score_bonus_scale` in the config.
Other terminal shaping terms, such as timeout, completion, and concentration penalties, are set to zero in the reported experiments.

Student ability is injected directly through `theta`:

| Student level | Theta |
| --- | ---: |
| low | `1.0` |
| mid | `2.0` |
| high | `3.0` |

Each student level corresponds to one ability value, and that value is used in the confidence equation above.

## 📝 Project Structure

```text
.
├── agents/        # PPO, DQN training logic and heuristic baselines
├── analysis/      # evaluation, comparison, trajectory inspection
├── configs/       # experiment configs for PPO and DQN
├── data/          # exam datasets and student profiles
├── env/           # exam environment, dynamics, reward, state
├── results/       # saved evaluation outputs
├── runs/          # trained model checkpoints and logs
├── tests/         # unit tests
└── utils/         # config loading, seed setup, compatibility helpers
```

Key entrypoints:

- `main.py`: train, eval, heuristic, and cross-validation modes
- `agents/train_rl.py`: PPO and DQN training
- `analysis/trajectory_report.py`: trajectory inspection

## 🗂️ Data

The dataset covers three mathematics subjects: calculus, geometry, and probability/statistics.
For each subject, the training set consists of six mock exams, and the final evaluation is performed on one unseen CSAT exam.

The dataset construction was prepared with reference to materials provided by [MegaStudy](https://www.megastudy.net/Entinfo/correctRate/main.asp?SubMainType=I&mOne=ipsi&mTwo=588).

| Subject | Training exams | Zero-shot CSAT eval |
| --- | --- | --- |
| `calculus` | `data/calculus/*.json` | `data/25_math_calculus.json` |
| `geometry` | `data/geometry/*.json` | `data/25_math_geometry.json` |
| `prob_stat` | `data/prob_stat/*.json` | `data/25_math_prob_stat.json` |

The final PPO/DQN configs shuffle problem order on reset to reduce memorization of fixed problem positions. Per-problem evaluation outputs and exported CSVs are therefore mapped back to the original problem id (`pid`).

## 🏋️ How to Run

Use the placeholders below:

- `<algo>`: `ppo` or `dqn`
- `<subject>`: `calculus`, `geometry`, or `prob_stat`
- `<level>`: `low`, `mid`, or `high`
- `<RUN_NAME>`: generated run directory name such as `ppo_20260421_161814` or `dqn_20260422_062908`

### Train

Direct command:

```bash
python main.py --mode train \
  --config configs/<algo>/<subject>/<level>.yaml \
  --output runs/<algo>/<subject>/<level>
```

Shell script:

```bash
bash scripts/train.sh <algo> <subject> <level>
```

Example:

```bash
bash scripts/train.sh ppo calculus mid
bash scripts/train.sh dqn calculus mid
```

### Evaluate

Evaluation is zero-shot: the trained model is evaluated on the final CSAT data file for the same subject.

Direct command:

```bash
python main.py --mode eval \
  --config runs/<algo>/<subject>/<level>/<RUN_NAME>/config_snapshot.yaml \
  --model-path runs/<algo>/<subject>/<level>/<RUN_NAME>/checkpoints/<algo>_final.zip \
  --algorithm <algo> \
  --exam-data data/25_math_<subject>.json \
  --episodes 100 \
  --output runs/zero_shot/<algo>/<subject>/<level>
```

For `prob_stat`, the CSAT file is `data/25_math_prob_stat.json`.

Shell script:

```bash
bash scripts/eval.sh <algo> <subject> <level>
```

This script automatically finds the most recent run under `runs/<algo>/<subject>/<level>/` and evaluates it on the matching CSAT file.

### Generate a Trajectory Report

Direct command:

```bash
python analysis/trajectory_report.py \
  --run-dir runs/<algo>/<subject>/<level>/<RUN_NAME> \
  --model-path runs/<algo>/<subject>/<level>/<RUN_NAME>/checkpoints/<algo>_final.zip \
  --algorithm <algo> \
  --exam-data data/25_math_<subject>.json \
  --episodes 10 \
  --max-logged-steps 80 \
  --output results/<algo>/<subject>/<level>_trajectory.json
```

Shell script:

```bash
bash scripts/traj.sh <algo> <subject> <level>
```

This script automatically finds the most recent run under `runs/<algo>/<subject>/<level>/` and generates a trajectory report on the matching CSAT file.

### Export Per-Problem Time Allocation CSVs

After zero-shot evaluation, export each model's average time allocation per problem:

```bash
python scripts/export_zero_shot_problem_times.py --algorithm ppo
python scripts/export_zero_shot_problem_times.py --algorithm dqn
```

The script reads `runs/zero_shot/<algo>/<subject>/<level>/eval_<algo>_*/rl_eval_detailed.json` and saves one CSV per subject-level pair:

```text
results/<algo>_zero_shot_problem_times/<subject>_<level>.csv
```

Each CSV uses the fixed columns:

```text
pid,difficulty,type,score,error_rate,avg_time_sec
```

To export only selected subjects or levels:

```bash
python scripts/export_zero_shot_problem_times.py \
  --algorithm dqn \
  --subjects calculus geometry prob_stat \
  --levels low mid high
```


## 📋 Results

### 0. Analytic Baseline

| Ability ($\theta$) | prob_stat | calculus | geometry |
| --- | ---: | ---: | ---: |
| low ($\theta=1.0$) | 56.93 | 69.62 | 66.89 |
| mid ($\theta=2.0$) | 74.71 | 84.69 | 82.80 |
| high ($\theta=3.0$) | 87.53 | 93.30 | 92.31 |

`Score Change vs Baseline` in the RL result tables is computed as `(Mean Score - Analytic Baseline) / Analytic Baseline`.
The primary metric is `Mean Score` on the unseen CSAT exam; `Mean Reward` includes the training reward definition and terminal bonus, so it is reported as a diagnostic rather than a direct score.

### 1. PPO

PPO models were trained on the six mock exams for each subject and evaluated zero-shot on the unseen CSAT exam.

| Subject | Level | Mean Score | Mean Reward | Mean Solved Count | Mean Coverage | Score Change vs Baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| calculus | low | 73.9833 | 54.3231 | 26.93 | 1.0000 | +6.27% |
| calculus | mid | 87.3845 | 67.8584 | 30.00 | 1.0000 | +3.18% |
| calculus | high | 94.4262 | 74.9705 | 30.00 | 1.0000 | +1.21% |
| geometry | low | 71.6399 | 51.9563 | 25.80 | 1.0000 | +7.10% |
| geometry | mid | 85.8228 | 66.2810 | 30.00 | 1.0000 | +3.65% |
| geometry | high | 93.7543 | 74.2918 | 30.00 | 1.0000 | +1.56% |
| prob_stat | low | 63.5932 | 43.8291 | 24.33 | 0.9997 | +11.70% |
| prob_stat | mid | 80.6065 | 61.0125 | 29.98 | 1.0000 | +7.89% |
| prob_stat | high | 91.1757 | 71.6874 | 30.00 | 1.0000 | +4.17% |

### 2. DQN

DQN models were trained on the same six mock exams and evaluated zero-shot on the unseen CSAT exam.

| Subject | Level | Mean Score | Mean Reward | Mean Solved Count | Mean Coverage | Score Change vs Baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| calculus | low | 74.3402 | 54.6836 | 28.25 | 1.0000 | +6.78% |
| calculus | mid | 86.8490 | 67.3175 | 29.99 | 1.0000 | +2.55% |
| calculus | high | 93.8314 | 74.3698 | 29.98 | 0.9993 | +0.57% |
| geometry | low | 71.9308 | 52.2501 | 27.02 | 1.0000 | +7.54% |
| geometry | mid | 85.8593 | 66.3179 | 30.00 | 1.0000 | +3.69% |
| geometry | high | 93.7143 | 74.2515 | 30.00 | 1.0000 | +1.52% |
| prob_stat | low | 64.1232 | 44.3644 | 24.69 | 0.9913 | +12.64% |
| prob_stat | mid | 80.6617 | 61.0683 | 29.95 | 1.0000 | +7.97% |
| prob_stat | high | 91.0286 | 71.5389 | 30.00 | 1.0000 | +4.00% |

## Team Contributions

- `Jinwoong Jung`: RL environment design, confidence score modeling, reward function design, PPO/DQN implementation, and experiments
- `Yuji Lim`: Presentation slide design, project report writing, result analysis, and visualization
- `Sangyun Lee`: Project report writing, presentation slide design, and related work survey
- `Jungwoo Choi`: Presentation slide design, baseline experiments, data collection, and preprocessing

## Acknowledgement

This project was conducted as part of the Sungkyunkwan University course `AAI2024 - Introduction to Reinforcement Learning`.
We would like to acknowledge [Jaekwang KIM](https://scholar.google.com/citations?hl=ko&user=zyjtJZwAAAAJ), Associate Professor at Sungkyunkwan University, for the course and academic guidance that supported this work.
