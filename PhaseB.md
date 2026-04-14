# Phase B

Phase B의 목표는 `Phase A`에서 확정한 환경 위에서 강화학습 정책을 실제로 학습시키고, 그 정책이 heuristic baseline보다 나은지 또는 최소한 어느 수준까지 따라가는지를 검증하는 것이다. 이번 프로젝트의 주장은 “하나의 범용 정책”이 아니라 “학생 능력대별 맞춤 정책 추천”이므로, Phase B는 `low`, `mid`, `high` 각각에 대해 별도 정책을 학습하고 비교하는 방향으로 진행한다.

## 1. Phase B에서 고정하는 기본 환경

Phase A 결과를 반영한 기본 환경은 다음과 같다.

- `switch_time_sec = 10`
- `action_time_unit_sec = 60`
- `anchor_source = "correct_rate"`
- `ambiguity_weight = 0.5`
- `subjective_floor = 0.0`
- 학생 능력은 `low / mid / high`로 분리

이 기준 설정은 [`configs/phaseB/base.yaml`](/Users/jinwoong/RLProject/configs/phaseB/base.yaml:1)에 정리되어 있다. 실제 학습용 config는 다음 세 파일을 사용한다.

- [`configs/phaseB/train_low.yaml`](/Users/jinwoong/RLProject/configs/phaseB/train_low.yaml:1)
- [`configs/phaseB/train_mid.yaml`](/Users/jinwoong/RLProject/configs/phaseB/train_mid.yaml:1)
- [`configs/phaseB/train_high.yaml`](/Users/jinwoong/RLProject/configs/phaseB/train_high.yaml:1)

이 세 파일은 동일한 환경을 공유하고, 차이는 `student.fixed_level`과 `experiment.name`뿐이다.

## 2. Phase B의 핵심 질문

Phase B는 아래 질문에 답하는 단계다.

1. PPO가 각 능력대에서 강한 heuristic baseline인 `marginal_gain_greedy`를 넘을 수 있는가?
2. `low`, `mid`, `high`에서 각각 따로 학습한 정책이 실제로 필요한가?
3. 어떤 능력대에서 학습한 정책을 다른 능력대에 적용하면 성능이 얼마나 떨어지는가?
4. reward shaping이 실제로 학습에 도움이 되는가?
5. PPO와 DQN 중 어떤 알고리즘이 이 문제에 더 적합한가?

즉, Phase B는 단순히 “RL이 되느냐”만 보는 것이 아니라, “어떤 RL 설정이 어떤 학생군에서 의미가 있느냐”를 정리하는 단계다.

## 3. 권장 실행 순서

Phase B는 아래 순서로 진행하는 것이 가장 안정적이다.

1. `mid` 학생용 PPO를 먼저 학습해 전체 파이프라인이 정상 동작하는지 확인한다.
2. `low`, `high`로 확장해 능력대별 PPO 3개를 확보한다.
3. 각 정책을 자기 능력대에서 평가한다.
4. 각 정책을 다른 능력대에도 적용해 교차 평가한다.
5. PPO가 기준선을 형성한 뒤 reward ablation을 수행한다.
6. 마지막으로 DQN을 비교 알고리즘으로 추가한다.

처음부터 PPO, DQN, reward ablation을 한꺼번에 돌리기보다, `PPO same-level -> PPO cross-level -> reward -> DQN` 순서로 가는 것이 해석과 디버깅 모두에 유리하다.

## 4. Step 1: PPO same-level 학습

가장 먼저 각 능력대별 PPO 정책을 학습한다. multi-seed 학습을 기본으로 사용해 seed 편차를 함께 본다. 현재 학습 스크립트는 [`agents/train_rl.py`](/Users/jinwoong/RLProject/agents/train_rl.py:675)에서 `--config`, `--output`, `--seeds`를 지원한다.

### 4.1 Mid 먼저 확인

```bash
python agents/train_rl.py \
  --config configs/phaseB/train_mid.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/train_mid
```

이 단계의 목적은 다음 두 가지다.

- PPO 학습이 정상적으로 끝나는지 확인
- 최종 score가 heuristic baseline과 비교할 가치가 있는 수준인지 확인

### 4.2 전체 능력대 PPO 학습

```bash
python agents/train_rl.py \
  --config configs/phaseB/train_low.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/train_low

python agents/train_rl.py \
  --config configs/phaseB/train_mid.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/train_mid

python agents/train_rl.py \
  --config configs/phaseB/train_high.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/train_high
```

각 실행이 끝나면 `runs/phaseB/...` 아래에 다음이 저장된다.

- seed별 run 디렉터리
- `checkpoints/ppo_final.zip`
- `eval/ppo_eval.json`
- `config_snapshot.yaml`
- multi-seed summary JSON

## 5. Step 2: PPO same-level 평가

학습이 끝났으면 각 정책을 자기 능력대에서 heuristic들과 같은 표 위에서 비교한다. 이때 [`analysis/run_comparison.py`](/Users/jinwoong/RLProject/analysis/run_comparison.py:158)를 사용하면 heuristic과 RL 모델이 한 표에 같이 들어간다.

```bash
python analysis/run_comparison.py \
  --config configs/phaseB/train_low.yaml \
  --runs-dir runs/phaseB/train_low \
  --student-level low \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/same_level/low

python analysis/run_comparison.py \
  --config configs/phaseB/train_mid.yaml \
  --runs-dir runs/phaseB/train_mid \
  --student-level mid \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/same_level/mid

python analysis/run_comparison.py \
  --config configs/phaseB/train_high.yaml \
  --runs-dir runs/phaseB/train_high \
  --student-level high \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/same_level/high
```

여기서 가장 먼저 볼 지표는 다음과 같다.

- `Real.Score`
- `Reward`
- `Coverage`
- `Solved`
- `Steps`

특히 `marginal_gain_greedy`와 RL의 `Real.Score` 차이가 핵심이다. PPO가 heuristic을 넘지 못하더라도 충분히 근접하고 행동 패턴이 자연스럽다면, 이후 reward shaping이나 algorithm 변경의 가치가 생긴다.

## 6. Step 3: PPO cross-level 평가

이번 프로젝트의 핵심 주장은 “능력대별 맞춤 정책 추천”이므로, same-level 평가만으로는 부족하다. 반드시 교차 평가를 통해 “자기 레벨에서 학습한 정책이 다른 레벨에서는 덜 맞는다”는 점을 확인해야 한다.

예를 들어 `low` 정책을 `mid`, `high`에 적용하는 방법은 다음과 같다.

```bash
python analysis/run_comparison.py \
  --config configs/phaseB/train_low.yaml \
  --runs-dir runs/phaseB/train_low \
  --student-level mid \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/cross_level/low_to_mid

python analysis/run_comparison.py \
  --config configs/phaseB/train_low.yaml \
  --runs-dir runs/phaseB/train_low \
  --student-level high \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/cross_level/low_to_high
```

같은 방식으로 `mid -> low/high`, `high -> low/mid`도 모두 평가한다.

정리하면 다음 6개를 추가로 돌리면 된다.

- `low -> mid`
- `low -> high`
- `mid -> low`
- `mid -> high`
- `high -> low`
- `high -> mid`

교차 평가의 목적은 다음과 같다.

- 같은 능력대에서 학습한 정책이 자기 환경에서 가장 좋은지 확인
- 다른 능력대에 적용했을 때 성능이 얼마나 떨어지는지 확인
- “범용 정책”보다 “맞춤 정책”이 더 타당하다는 근거 확보

## 7. Step 4: Reward ablation

Reward ablation은 heuristic 단계가 아니라 RL 단계에서만 의미가 있다. 현재 기본 reward는 pure reward이고, shaping 버전은 별도 config로 제공된다.

- [`configs/phaseB/reward/reward_pure.yaml`](/Users/jinwoong/RLProject/configs/phaseB/reward/reward_pure.yaml:1)
- [`configs/phaseB/reward/reward_shaping.yaml`](/Users/jinwoong/RLProject/configs/phaseB/reward/reward_shaping.yaml:1)

가장 먼저 `mid`에서 reward 차이를 보는 것이 실용적이다.

```bash
python agents/train_rl.py \
  --config configs/phaseB/reward/reward_pure.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/reward_pure

python agents/train_rl.py \
  --config configs/phaseB/reward/reward_shaping.yaml \
  --seeds 42,123,2024 \
  --output runs/phaseB/reward_shaping
```

그다음 같은 조건에서 비교 평가한다.

```bash
python analysis/run_comparison.py \
  --config configs/phaseB/reward/reward_pure.yaml \
  --runs-dir runs/phaseB/reward_pure \
  --student-level mid \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/reward/pure_mid

python analysis/run_comparison.py \
  --config configs/phaseB/reward/reward_shaping.yaml \
  --runs-dir runs/phaseB/reward_shaping \
  --student-level mid \
  --episodes 60 \
  --realized-rollouts 100 \
  --output results/phaseB/reward/shaping_mid
```

이 단계에서는 다음을 확인한다.

- shaping이 `Real.Score`를 올리는가
- shaping이 `Coverage`를 안정화하는가
- shaping이 특정 문제에 시간 쏠림을 줄이는가

reward ablation은 기본 PPO 파이프라인이 어느 정도 안정화된 뒤에 진행하는 것이 좋다.

## 8. Step 5: DQN 비교

PPO 결과가 확보되면, 같은 환경에서 DQN도 비교 알고리즘으로 돌린다. DQN은 [`agents/train_rl.py`](/Users/jinwoong/RLProject/agents/train_rl.py:530) 내부에서 `DiscreteActionWrapper`를 거쳐 학습된다.

가장 현실적인 순서는 다음과 같다.

1. `mid`에서 DQN 3-seed 실행
2. PPO와 같은 방식으로 same-level 평가
3. PPO와 차이가 충분히 의미 있으면 `low`, `high`로 확장

실행 자체는 config의 `training.algorithm`을 `dqn`으로 바꾼 파일이 필요하다. 만약 Phase B에서 DQN까지 본격 비교할 계획이라면 `train_mid_dqn.yaml` 같은 별도 config를 추가하는 것이 가장 깔끔하다.

## 9. RL 결과를 어떻게 해석할 것인가

Phase B에서 RL을 해석할 때는 단순히 `mean_score` 하나만 보면 안 된다. 최소한 아래 지표를 같이 봐야 한다.

- `Real.Score`: 실제 최종 성능
- `Reward`: 학습 목적함수와의 일치 여부
- `Coverage`: 전체 문제를 얼마나 폭넓게 보았는지
- `Solved`: 실제로 풀린 문제 수
- `Steps`: 의사결정 빈도

추가로 학습 후 평가 JSON에는 다음도 함께 확인하는 것이 좋다.

- `mean_top1_time_share`
- `mean_top2_time_share`
- `mean_objective_dominance_rate`
- `mean_subjective_confidence`
- `mean_subjective_solved_rate`
- `mean_objective_solved_rate`

특히 `top1/top2 time share`는 “정책이 소수 문제에 과도하게 집착하는가”를 보여주는 핵심 지표다. score가 높아 보여도 상위 1~2문항에 시간을 몰아 쓰는 정책이면 일반적인 시험 전략으로 해석하기 어렵다.

## 10. Trajectory sanity check

숫자만으로 정책 해석이 어려울 때는 trajectory를 직접 확인한다. 이때 [`analysis/trajectory_report.py`](/Users/jinwoong/RLProject/analysis/trajectory_report.py:1)를 사용한다.

이 스크립트로 확인할 항목은 다음과 같다.

- 실제 방문 순서
- 문제별 사용 시간
- revisit가 의미 있게 일어났는지
- 상위 1~2문제에 시간 집중이 있는지
- objective / subjective 문제에서 confidence가 어떻게 올라가는지

즉, same-level이나 cross-level 결과가 이상할 때는 trajectory report로 “점수는 왜 이렇게 나왔는가”를 행동 단위에서 확인한다.

## 11. 최종 산출물 형태

Phase B가 끝나면 최소한 아래 산출물이 있어야 한다.

- `runs/phaseB/train_low`, `train_mid`, `train_high`
- 각 seed별 PPO run
- 각 능력대 same-level 비교표
- 각 능력대 cross-level 비교표
- reward ablation 결과
- 가능하면 PPO vs DQN 비교표

결과 정리 표는 아래 구조로 모으면 좋다.

1. Same-level PPO vs heuristic
2. Cross-level PPO transfer
3. Reward pure vs shaping
4. PPO vs DQN

## 12. 권장 마일스톤

가장 현실적인 진행 순서는 다음과 같다.

### Milestone 1

- `train_mid.yaml`로 PPO 3-seed 학습
- `mid` same-level 평가
- heuristic 대비 성능 확인

### Milestone 2

- `train_low.yaml`, `train_high.yaml` PPO 3-seed 학습
- `low`, `high` same-level 평가
- 능력대별 정책 차이 확인

### Milestone 3

- 6개 cross-level 평가 실행
- 맞춤 정책 주장 검증

### Milestone 4

- reward pure vs shaping 비교
- 필요하면 DQN 추가

## 13. 최종 목표

Phase B의 최종 목적은 단순히 “RL이 heuristic보다 좋은가”를 보여주는 데서 끝나지 않는다. 더 중요한 목표는 `low`, `mid`, `high` 학생군이 서로 다른 시간 배분 전략을 필요로 하며, 따라서 하나의 범용 정책보다 능력대별 맞춤 정책 추천이 더 설득력 있는 접근이라는 점을 실험적으로 보여주는 것이다.
