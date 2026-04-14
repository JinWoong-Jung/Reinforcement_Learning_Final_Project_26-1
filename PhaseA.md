Ablation 실행 전략: 2단계로 나눠야 합니다

Phase A (지금 당장 실행 가능 — 학습 없음)
  run_comparison.py로 각 ablation config에서 heuristic 점수 비교

Phase B (RL 학습 후 실행)
  각 ablation config로 RL 학습 → default RL과 비교
이 두 단계가 답할 질문이 다릅니다.
	Phase A (heuristic)	Phase B (RL)
목적	환경 자체의 특성 이해	RL 설계 선택의 영향 측정
시간	몇 분	수 시간
먼저 해야 하는 이유	환경이 이상하면 RL을 아무리 돌려도 의미없음	heuristic 기준선 없이 RL 결과를 해석 불가
Phase A — 지금 실행해야 할 명령들
Group 1. Switch cost ablation (가장 먼저)

# 4개 config를 순서대로 또는 병렬로 실행
for cost in 0 5 10 20; do
  if [ $cost -eq 10 ]; then
    cfg="configs/default.yaml"
  else
    cfg="configs/ablation/switch_cost_${cost}s.yaml"
  fi
  python analysis/run_comparison.py \
    --config $cfg \
    --episodes 50 \
    --realized-rollouts 100 \
    --output results/ablation/switch_cost_${cost}s/
done
해석 방법:
switch_cost가 커질수록:
* marginal_gain_greedy 점수가 얼마나 떨어지는가? → greedy가 전환 비용에 얼마나 민감한지
* random과 greedy의 격차가 커지는가 줄어드는가? → 전환 비용이 클수록 "신중한 전략"이 더 유리해야 함
* mean_steps가 줄어드는가? → 높은 전환 비용이 플레이어를 한 문제에 오래 머물게 만드는지 확인
기대 패턴: switch_cost=0 → 가장 높은 점수 + 가장 많은 전환. switch_cost=20 → 점수 하락, 문제당 시간 집중.
만약 패턴이 반대라면 dynamics 모델을 재검토해야 합니다.

Group 2. Action granularity ablation

for unit in 10 30 60; do
  if [ $unit -eq 30 ]; then
    cfg="configs/default.yaml"
  else
    cfg="configs/ablation/action_${unit}s.yaml"
  fi
  python analysis/run_comparison.py \
    --config $cfg \
    --episodes 50 \
    --realized-rollouts 100 \
    --output results/ablation/action_${unit}s/
done
해석 방법:
* marginal_gain_greedy 점수 변화: 10초 단위가 30초보다 유의미하게 높은가?
    * 높다 → 세밀한 시간 배분이 실제로 중요함 → RL이 action_10s에서 더 큰 이득을 볼 수 있음
    * 거의 같다 → 30초 단위로 충분, 10초의 복잡도 증가 비용이 없음
* mean_steps 비율 확인: action_10s의 steps ≈ action_30s × 3 이어야 정상
핵심 질문: greedy@10s > greedy@30s > greedy@60s 인가? 만약 그렇다면 RL도 같은 패턴을 따르는지 Phase B에서 검증합니다.

Group 3. Student skill ablation

for level in fixed_mid mixed; do
  python analysis/run_comparison.py \
    --config configs/ablation/student_${level}.yaml \
    --episodes 60 \
    --realized-rollouts 100 \
    --output results/ablation/student_${level}/
done
해석 방법:
* student_level_breakdown (per-level 분해)를 --episodes 60으로 충분히 모아서 확인
    * student_fixed_mid 결과: mid 레벨에서 점수가 student_mixed보다 높은가?
    * student_mixed 결과: low/mid/high 각각의 greedy 점수 차이가 크면 → 학생 능력이 정책에 중요한 조건임
* 두 config의 greedy 점수 차이가 크면: RL 학습 때 어떤 학생 분포를 쓸지가 중요한 설계 선택

Group 4. Probability model ablation

for variant in no_ambiguity no_subjective_floor anchor_difficulty; do
  python analysis/run_comparison.py \
    --config configs/ablation/prob_${variant}.yaml \
    --episodes 50 \
    --realized-rollouts 100 \
    --output results/ablation/prob_${variant}/
done
해석 방법:
config	기준 대비 greedy 점수 변화	의미
no_ambiguity	거의 같음	choice entropy가 실제로 별 영향 없음 → 제거 가능
no_ambiguity	눈에 띄게 낮음	ambiguity가 실제로 어려운 문제를 분류하는 데 중요
no_subjective_floor	주관식 점수 낮아짐	floor=0.02가 현실적임을 확인
anchor_difficulty	거의 같음	difficulty 필드 ≈ 1-correct_rate (중복 feature)
anchor_difficulty	차이 남	두 feature가 다른 정보 담고 있음
이 ablation은 "우리 P_i(t) 모델이 어떤 feature에 실제로 민감한가"를 측정합니다. heuristic 단계에서도 충분히 드러납니다.

Reward ablation은 Phase B에서만 의미 있음
reward_pure vs reward_shaping은 heuristic으로는 의미없습니다 — heuristic은 reward를 학습에 사용하지 않으니까요. 이건 RL 학습 후에 비교합니다.

# Phase B (RL 학습 후)
for variant in pure shaping; do
  python agents/train_rl.py \
    --config configs/ablation/reward_${variant}.yaml \
    --seeds 42,123,2024 \
    --output runs/ablation/

  python analysis/run_comparison.py \
    --config configs/ablation/reward_${variant}.yaml \
    --runs-dir runs/ablation/ \
    --episodes 60 \
    --output results/ablation/reward_${variant}/
done

전체 실행 순서 (권장)

지금 실행 (Phase A — 30분~1시간)
──────────────────────────────────────────────
① switch cost 4종   → environment 민감도 파악
② action granularity 3종 → 적절한 dt 선택
③ probability model 3종  → dynamics feature 검증
④ student 2종       → 학생 분포 영향 파악

RL 학습 (Phase B — 수 시간)
──────────────────────────────────────────────
⑤ default config로 PPO 3-seed 학습 (기준점)
⑥ reward_pure / reward_shaping 학습 후 비교
⑦ 가장 좋은 설정으로 DQN도 학습 후 PPO와 비교
⑧ run_comparison.py로 최종 표 생성

Phase A 결과를 보고 결정할 것들
Phase A가 끝나면 다음 질문에 답할 수 있어야 RL 학습 설정이 확정됩니다:
1. switch_time_sec 값: 0s~20s 중 어느 값이 "도전적이지만 학습 가능"한가 → 너무 작으면 trivial, 너무 크면 coverage 불가
2. action_time_unit_sec: 10s가 30s보다 greedy 점수가 크게 높으면 10s로 학습, 비슷하면 30s (학습 비용 절감)
3. ambiguity_weight: greedy가 no_ambiguity에서 별로 안 떨어지면 0으로 고정해 모델 단순화
4. 학생 설정: 실험의 주장이 "특정 학생"인지 "일반 학생"인지에 따라 fixed_mid vs mixed 선택
이 4가지가 결정되면 그것이 RL 학습의 기본 config가 됩니다. 그 위에서 reward_pure vs reward_shaping을 비교하는 것이 올바른 ablation 순서입니다.