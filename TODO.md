# RL 시험시간 배분 프로젝트 구현 체크리스트

## 0. 프로젝트 목표를 한 줄로 고정

먼저 이것부터 문서 상단에 명시하세요.

* **목표:** 제한 시간 내 총 기대점수 또는 총 실현점수를 최대화하는 문제 풀이 전략 학습
* **전략:**

  1. 어떤 문제를 다음에 풀지
  2. 각 문제에 시간을 얼마나 더 투입할지 결정

이 문장이 흔들리면 나머지도 흔들립니다.

---

## 1. 문제 정의 재정리

### 1-1. 상태(state) 정의 점검

확인할 것:

* 남은 총 시간
* 현재 선택 중인 문항
* 각 문항의 현재 상태

  * not_started
  * in_progress
  * finished
  * skipped
* 각 문항에 누적 투입된 시간
* 각 문항의 배점
* 각 문항의 유형
* 각 문항의 difficulty 계열 feature
* 현재까지 누적 기대점수 또는 누적 실현점수

주의:

* 정답확률의 내부 latent 값은 state에 직접 노출하지 않는 편이 더 자연스럽습니다.
* 하지만 **정답확률 계산에 필요한 observable feature는 충분히 state에 들어 있어야** 합니다.

### 1-2. 행동(action) 정의 점검

행동은 너무 복잡하면 안 됩니다.

권장:

* `select_next_problem`
* `allocate_delta_time`

구체적으로는 둘 중 하나가 좋습니다.

#### 방식 A: 분리형

* 다음에 풀 문제 index 선택
* 시간 추가량 선택: 10초, 20초, 30초 등

#### 방식 B: 단일형

* `(problem_id, delta_t)` 조합 action

처음엔 **작은 discrete action space**가 좋습니다.

### 1-3. 종료조건(done) 정의

* 총 시간 소진
* 모든 문제가 종료 상태
* 더 이상 행동할 가치가 없는 상태

---

## 2. 정답확률 모델 설계 체크

이 부분이 핵심입니다.

## 2-1. 각 문항의 정답확률 함수 (P_i(t)) 명시

반드시 식으로 적어두세요.

예시:
[
P_i(t)=c_i+(1-c_i)\cdot \sigma(\alpha_s+\beta \log(1+t)-\gamma d_i-\delta a_i)
]

여기서 최소한 아래는 정해야 합니다.

* (c_i): chance floor
* (d_i): difficulty 관련 값
* (a_i): ambiguity 관련 값
* 학생 능력 파라미터 포함 여부

## 2-2. feature 사용 기준 정리

각 feature를 왜 쓰는지 명확히 적으세요.

* `difficulty`: 문항 난이도
* `correct_rate`: 집단 수준 난이도 proxy
* `error_rate`: 혼동도 또는 난이도 proxy
* `choice_rate`: objective 문항 ambiguity
* `question_type`: objective / subjective 차이 반영

중요:

* `difficulty`, `correct_rate`, `error_rate`를 **다 같이 막 넣지 말고**, 어떤 의미로 쓰는지 정리해야 합니다.
* 서로 거의 같은 축이면 하나만 쓰거나 역할을 분리하세요.

## 2-3. 객관식 chance floor 정의

체크:

* 단순히 (1/K)로 둘 것인지
* 보기 선택비율을 활용할 것인지
* choice_rate는 floor가 아니라 ambiguity feature로만 쓸 것인지

권장:

* objective는 기본 floor (1/K)
* choice distribution은 entropy/top-2 gap feature로 활용

## 2-4. 주관식 확률 정의

체크:

* 0으로 둘지
* 작은 epsilon을 줄지
* 시간이 늘어남에 따라 ceiling이 objective보다 낮게 갈지

---

## 3. 보상함수(reward) 체크

## 3-1. 기본 reward를 단순화

우선 실험의 기본 reward는 아래처럼 두세요.

[
r_t = \Delta \left(\sum_i w_i P_i(t_i)\right)
]

즉,

* 총 기대점수 증가량

## 3-2. shaping 제거 또는 최소화

체크:

* streak bonus 제거
* arbitrary penalty 제거
* coverage bonus 제거
* hand-crafted heuristic reward 제거

정말 필요한 것만 남기세요.

## 3-3. 이동 비용 / 전환 비용 처리

전환 비용은 reward shaping보다 **환경 dynamics**로 넣는 것이 더 좋습니다.

예:

* 문제 바꾸면 10초 차감
* 다시 돌아와도 같은 비용 적용 여부 명시

---

## 4. 환경(env) sanity check

## 4-1. 단일 문항 curve 시각화

각 문항에 대해 확인:

* time vs (P_i(t))
* time vs expected score
* marginal gain vs time

확인 포인트:

* 쉬운 문제는 초반 상승이 가파른가
* 어려운 문제는 늦게 상승하는가
* 배점이 높은 문제는 시간이 충분할 때 가치가 커지는가
* subjective/objective 차이가 반영되는가

## 4-2. 수작업 시뮬레이션

몇 개의 toy 문항에 대해 손으로 계산해 보세요.

예:

* 문항 3개
* 총 시간 100초
* switch cost 10초

이때 직관적인 최적 전략과 env 결과가 비슷해야 합니다.

## 4-3. edge case 체크

* 시간이 0일 때
* 문항 1개만 있을 때
* 모든 문항이 동일할 때
* 전부 objective일 때
* 전부 subjective일 때
* 배점은 높은데 확률 상승이 거의 없는 문제

---

## 5. baseline 구현 체크

RL 전에 반드시 구현해야 합니다.

## 5-1. Random baseline

* 랜덤 문제 선택
* 랜덤 시간 배분

## 5-2. Sequential baseline

* 1번부터 끝까지 순서대로
* 균등 시간 배분

## 5-3. Easy-first baseline

* 쉬운 문제부터 풀이
* 남는 시간 뒤에 배분

## 5-4. High-score-first baseline

* 배점 높은 문제부터 풀이

## 5-5. Greedy marginal gain baseline

가장 중요합니다.

매 step마다
[
\arg\max_i \frac{\Delta(w_i P_i)}{\Delta t + \text{switch cost}}
]
를 선택

이건 반드시 넣어야 합니다.

## 5-6. Optional planning baseline

가능하면 하나 더:

* beam search
* DP 근사
* rollout search

---

## 6. RL 학습 세팅 체크

## 6-1. 작은 toy env 먼저

바로 full 30문항 가지 말 것.

순서:

* 3문항
* 5문항
* 10문항
* 30문항

## 6-2. 알고리즘 최소 2개 비교

예:

* PPO
* DQN

단, action space가 복합적이면 PPO가 더 자연스러울 수 있습니다.

## 6-3. observation normalization

체크:

* 시간 값 스케일링
* 배점 정규화
* difficulty 정규화
* question type one-hot 처리 여부

## 6-4. exploration 설계

DQN이면:

* epsilon decay

PPO면:

* entropy coefficient

확인 포인트:

* 초기에 충분히 다양한 문항 선택이 일어나는가
* 특정 문제에만 고착되지 않는가

## 6-5. reproducibility

반드시 고정:

* random seed
* dataset split seed
* env generation seed
* model init seed

---

## 7. 평가 프로토콜 체크

## 7-1. expected score 평가

정책이 만든 시간 배분에 대해
[
\sum_i w_i P_i(t_i)
]
계산

## 7-2. realized score 평가

각 문항을 Bernoulli 샘플링해서 실제 맞고 틀림 생성 후 총점 계산

* 100회 rollout 평균
* 표준편차도 같이 기록

## 7-3. 정책 행동 해석

아래도 함께 저장하세요.

* 첫 선택 문항 분포
* 문항별 평균 배정 시간
* 난이도별 평균 배정 시간
* 문제 전환 횟수
* skip 비율
* objective vs subjective 시간 비율

## 7-4. baseline 대비 개선폭

반드시 표로 정리:

* Random
* Sequential
* Easy-first
* Score-first
* Greedy
* RL

---

## 8. ablation study 체크

최소한 아래는 해보는 것이 좋습니다.

## 8-1. probability model ablation

* choice_rate feature 사용 / 미사용
* subjective epsilon 사용 / 미사용
* correct_rate 사용 / 미사용

## 8-2. reward ablation

* pure expected score only
* shaping 추가 버전

## 8-3. switch cost ablation

* 0초
* 5초
* 10초
* 20초

## 8-4. student skill ablation

* 단일 학생 프로필
* 여러 학생 프로필 분포

## 8-5. action granularity ablation

* 10초 단위
* 30초 단위
* 60초 단위

---

## 9. “RL이 필요한가” 검증 체크

이 질문에는 꼭 답해야 합니다.

### 확인할 것

* greedy baseline이 이미 거의 최적인가
* planning baseline이 RL보다 더 강한가
* RL이 시험지 분포가 바뀌어도 일반화되는가
* unseen exam에서 RL이 heuristic보다 나은가

결론을 아래 둘 중 하나로 정리할 수 있어야 합니다.

* **이 문제는 planning이 더 적합하다**
* **이 문제는 exam distribution generalization 때문에 RL이 더 적합하다**

---

## 10. 결과 정리 체크

최종 보고서/발표에 최소한 들어가야 하는 것:

## 10-1. 문제 정의

* 상태
* 행동
* 보상
* 종료조건

## 10-2. 정답확률 모델

* 식
* feature 설명
* 왜 그렇게 정의했는지

## 10-3. baseline 비교표

* 각 방법별 점수
* RL 성능 비교

## 10-4. 전략 시각화

* 어떤 순서로 문제를 풀었는지
* 각 문제에 얼마나 시간을 썼는지
* difficulty별 시간 분배

## 10-5. 한계점

* 실제 학생 응답 데이터가 없어서 probability model이 synthetic
* 현재는 simulator-based optimization
* real-world validity는 추가 검증 필요

---

# 바로 실행용 TODO

아래부터 차례대로 하면 됩니다.

## Phase 1. 환경 정리

* [ ] 상태/행동/보상/종료조건 문서화
* [ ] 정답확률 함수 (P_i(t)) 고정
* [ ] 문항별 curve 시각화
* [ ] edge case 테스트

## Phase 2. baseline

* [ ] Random baseline 구현
* [ ] Sequential baseline 구현
* [ ] Easy-first baseline 구현
* [ ] Score-first baseline 구현
* [ ] Greedy marginal gain baseline 구현

## Phase 3. toy 검증

* [ ] 3문항 toy env 생성
* [ ] 5문항 toy env 생성
* [ ] baseline 결과 검증
* [ ] RL이 직관적 정책을 배우는지 확인

## Phase 4. RL 실험

* [ ] PPO 학습
* [ ] DQN 학습
* [ ] seed 3개 이상 반복
* [ ] learning curve 저장

## Phase 5. full-scale 평가

* [ ] 30문항 환경 평가
* [ ] expected score 측정
* [ ] realized score rollout 측정
* [ ] baseline 대비 개선폭 정리

## Phase 6. ablation

* [ ] reward ablation
* [ ] feature ablation
* [ ] switch cost ablation
* [ ] action granularity ablation

## Phase 7. 결과 정리

* [ ] 표 작성
* [ ] 전략 시각화
* [ ] 실패 사례 정리
* [ ] RL 필요성 결론 정리

---

# 가장 중요한 우선순위 5개만 뽑으면

1. **정답확률 함수부터 고정**
2. **reward를 단순화**
3. **greedy baseline 구현**
4. **toy env에서 sanity check**
5. **그 후에 RL 비교**

이 5개가 핵심입니다.
