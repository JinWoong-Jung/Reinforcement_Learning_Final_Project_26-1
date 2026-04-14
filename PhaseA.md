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
  cfg="configs/phaseA/group1_switch_cost/switch_cost_${cost}s.yaml"
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
  cfg="configs/phaseA/group2_action_granularity/action_${unit}s.yaml"
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

for level in low mid high; do
  python analysis/run_comparison.py \
    --config configs/phaseA/group3_student_skill/student_${level}.yaml \
    --episodes 60 \
    --realized-rollouts 100 \
    --output results/ablation/student_${level}/
done
해석 방법:
* `student_low`, `student_mid`, `student_high`의 greedy 점수 차이를 비교
    * low/mid/high 차이가 크면 → 학생 능력이 정책에 중요한 조건임
* 특히 greedy와 random/equal_time의 격차가 low에서 더 큰지 확인
    * 그렇다면 낮은 능력대에서 전략 최적화의 가치가 더 큼
* 이 결과를 바탕으로 RL 주장을 "범용 정책"이 아니라 "능력대별 맞춤 정책"으로 설계

Reward ablation은 Phase B에서만 의미 있음
reward_pure vs reward_shaping은 heuristic으로는 의미없습니다 — heuristic은 reward를 학습에 사용하지 않으니까요. 이건 RL 학습 후에 비교합니다.

# Phase B (RL 학습 후)
for variant in pure shaping; do
  python agents/train_rl.py \
    --config configs/phaseB/reward/reward_${variant}.yaml \
    --seeds 42,123,2024 \
    --output runs/ablation/

  python analysis/run_comparison.py \
    --config configs/phaseB/reward/reward_${variant}.yaml \
    --runs-dir runs/ablation/ \
    --episodes 60 \
    --output results/ablation/reward_${variant}/
done

전체 실행 순서 (권장)

지금 실행 (Phase A — 30분~1시간)
──────────────────────────────────────────────
① switch cost 4종   → environment 민감도 파악
② action granularity 3종 → 적절한 dt 선택
③ student 3종       → 학생 능력대 영향 파악

RL 학습 (Phase B — 수 시간)
──────────────────────────────────────────────
④ default config로 PPO 3-seed 학습 (기준점)
⑤ reward_pure / reward_shaping 학습 후 비교
⑥ 가장 좋은 설정으로 DQN도 학습 후 PPO와 비교
⑦ run_comparison.py로 최종 표 생성

Phase A 결과를 보고 결정할 것들
Phase A가 끝나면 다음 질문에 답할 수 있어야 RL 학습 설정이 확정됩니다:
1. switch_time_sec 값: 0s~20s 중 어느 값이 "도전적이지만 학습 가능"한가 → 너무 작으면 trivial, 너무 크면 coverage 불가
2. action_time_unit_sec: 10s가 30s보다 greedy 점수가 크게 높으면 10s로 학습, 비슷하면 30s (학습 비용 절감)
3. 학생 설정: 실험의 주장이 "범용 정책"인지 "능력대별 맞춤 정책"인지 결정
이 3가지가 결정되면 그것이 RL 학습의 기본 config가 됩니다. 그 위에서 reward_pure vs reward_shaping을 비교하는 것이 올바른 ablation 순서입니다.
