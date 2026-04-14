# KSAT Exam Strategy via Reinforcement Learning

**Authors**: 정진웅, 이상윤, 임유지, 최정우  
**Affiliation**: Sungkyunkwan University  
**Course**: 2026-1 Introduction to Reinforcement Learning `AAI2014_01` (김재광) Final Project


수능 시험 시간 배분 전략을 강화학습으로 최적화하는 프로젝트.
에이전트는 제한된 시험 시간 내에 어떤 문제에 얼마나 시간을 쓸지 결정해 기대 점수를 최대화한다.

## 설치

```bash
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

## 빠른 시작

### 휴리스틱 베이스라인 평가

```bash
python main.py --mode heuristic
python main.py --mode heuristic --config configs/default.yaml --episodes 200
```

### RL 학습

```bash
# 단일 시드
python main.py --mode train --config configs/default.yaml

# 멀티 시드 (통계적 신뢰도)
python agents/train_rl.py --config configs/default.yaml --seeds 42,123,2024
```

### RL 평가

```bash
python main.py --mode eval --model-path runs/ppo_final.zip --algorithm ppo
```

### 비교표 생성 (휴리스틱 + RL 전체)

```bash
python analysis/run_comparison.py --config configs/default.yaml --runs-dir runs/
```

## Ablation 실험

`configs/ablation/` 에 5개 축의 ablation config가 준비되어 있다.

| 축 | 파일 |
|---|---|
| 확률 모델 | `prob_no_ambiguity.yaml`, `prob_no_subjective_floor.yaml`, `prob_anchor_difficulty.yaml` |
| 보상 구조 | `reward_pure.yaml`, `reward_shaping.yaml` |
| 전환 비용 | `switch_cost_0s.yaml`, `switch_cost_5s.yaml`, `switch_cost_20s.yaml` |
| 학생 분포 | `student_fixed_mid.yaml`, `student_mixed.yaml` |
| 행동 단위 | `action_10s.yaml`, `action_60s.yaml` |

**Phase A — 휴리스틱으로 먼저 확인:**
```bash
python main.py --mode heuristic --config configs/ablation/switch_cost_0s.yaml
```

**Phase B — RL 학습 후 비교:**
```bash
python main.py --mode train --config configs/ablation/reward_shaping.yaml
python analysis/run_comparison.py --config configs/ablation/reward_shaping.yaml --runs-dir runs/
```

## 테스트

```bash
python -m pytest tests/ -v
```

## 설정 파일 주요 항목

`configs/default.yaml` 에서 조정 가능한 핵심 파라미터:

- `exam.total_time_sec`: 총 시험 시간
- `exam.action_time_unit_sec`: solve_more 1회당 소비 시간
- `exam.switch_time_sec`: 문제 전환 비용
- `dynamics.anchor_source`: 신뢰도 앵커 (`correct_rate` / `difficulty`)
- `training.algorithm`: `ppo` 또는 `dqn`
- `training.total_steps`: 학습 스텝 수
