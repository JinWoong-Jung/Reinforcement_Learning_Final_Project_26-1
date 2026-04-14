# KSAT Exam Strategy via Reinforcement Learning

**Authors**: 정진웅, 이상윤, 임유지, 최정우  
**Affiliation**: Sungkyunkwan University  
**Course**: 2026-1 Introduction to Reinforcement Learning `AAI2014_01` (김재광) Final Project


수능 시험 시간 배분 전략을 강화학습으로 최적화하는 프로젝트.
에이전트는 제한된 시험 시간 내에 어떤 문제에 얼마나 시간을 쓸지 결정해 기대 점수를 최대화한다.

## 설치

### `conda` 환경 사용

```bash
conda create -n rlproject python=3.11 -y
conda activate rlproject
pip install -r requirements.txt
```

## 프로젝트 구조

```
configs/          # 실험 설정 (default.yaml, ablation/)
data/             # 시험 문제 JSON, 학생 프리셋
env/              # 시험 환경 (Gymnasium)
agents/           # 휴리스틱 정책, RL 학습
analysis/         # 평가, 비교표 생성
tests/            # 엣지케이스 테스트
```

## Phase A Ablations

### 1. Switch Cost

문제 사이를 이동할 때 드는 시간 비용 `switch_time_sec`를 바꿔, 전환 비용이 전략 성능에 어떤 영향을 주는지 본다.

```bash
for cost in 0 5 10 20; do
  cfg="configs/phaseA/group1_switch_cost/switch_cost_${cost}s.yaml"
  python analysis/run_comparison.py \
    --config $cfg \
    --student-level mid \
    --episodes 50 \
    --realized-rollouts 100 \
    --output results/phaseA/group1_switch_cost/switch_cost_${cost}s/
done
```

결과: `switch_cost`가 커질수록 `marginal_gain_greedy`와 `random`의 격차가 커졌고, `10초`가 가장 균형적인 기본값으로 확인되었다.

### 2. Action Granularity

한 번의 `solve_more` 행동이 몇 초를 의미하는지 바꿔, 시간 제어 단위의 적절한 크기를 비교한다.

```bash
for unit in 10 30 60; do
  cfg="configs/phaseA/group2_action_granularity/action_${unit}s.yaml"
  python analysis/run_comparison.py \
    --config $cfg \
    --student-level mid \
    --episodes 50 \
    --realized-rollouts 100 \
    --output results/phaseA/group2_action_granularity/action_${unit}s/
done
```

결과: `10초` 단위의 미세 제어는 성능 이득이 거의 없었고, `60초`가 가장 높은 점수와 가장 적은 step 수를 보여 기본 시간 단위로 채택되었다.

### 3. Student Skill

학생 능력 수준을 `low / mid / high`로 고정해, 능력대에 따라 어떤 전략 차이가 나타나는지 본다.

```bash
for level in low mid high; do
  cfg="configs/phaseA/group3_student_skill/student_${level}.yaml"
  python analysis/run_comparison.py \
    --config $cfg \
    --episodes 60 \
    --realized-rollouts 100 \
    --output results/phaseA/group3_student_skill/student_${level}/
done
```

결과: 학생 능력이 높을수록 절대 점수는 상승했지만, `marginal_gain_greedy`의 상대적 이득은 `low`에서 가장 크게 나타났다. 따라서 이후 RL은 범용 정책보다 능력대별 맞춤 정책 추천 방향으로 진행한다.
