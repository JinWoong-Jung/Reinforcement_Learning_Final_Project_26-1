# RLProject

이 리포지토리는 제한된 시험 시간 안에서 어떤 문제를 얼마나 풀고 언제 다음 문제로 넘어갈지를 강화학습으로 학습하는 프로젝트다. 핵심 목표는 실제 정답을 직접 맞히는 것이 아니라, 각 문제에 시간을 추가로 투자했을 때 기대 점수(`expected score`)가 어떻게 증가하는지를 모델링하고, 전체 시험에서 기대 총점을 최대화하는 정책을 학습하는 것이다.

## 프로젝트 구조

### 1. 실행 진입점

- `main.py`
  - `train`, `eval`, `heuristic` 모드를 제공한다.
  - 설정 파일은 기본적으로 `configs/default.yaml`을 읽는다.
  - `--exam-data`, `--student-data`, `--student-preset`, `--student-id`, `--student-level`로 데이터와 학생 조건을 덮어쓸 수 있다.

### 2. 환경(`env/`)

- `env/exam_env.py`
  - Gymnasium 스타일의 `ExamStrategyEnv`를 구현한다.
  - 상태는 남은 시간, 현재 위치, 각 문항의 진행 상태/투입 시간과 공개 문항 메타데이터를 하나의 고정 길이 벡터로 반환한다.
  - confidence는 문항 타입에 따라 다르게 observation에 들어간다.
    - 주관식: `[answer_confidence, 0, 0, 0, 0]`
    - 객관식: `[c1, c2, c3, c4, c5]`
  - 행동은 `MultiDiscrete([2, 문항 수])`이며, 각 step에서
    - `action_type=0`: 현재 문항에 시간을 더 쓰기 (`solve_more`)
    - `action_type=1`: 다른 문항으로 이동하기 (`next`, 재방문 가능)
    를 의미한다.
  - `next`에도 별도의 이동 시간 비용이 들어가며, 현재 문제를 target으로 고르면 자동으로 다른 문제로 redirect된다.
  - 설정된 제약을 넘으면 `solve_more` 요청도 자동으로 `next`로 강제 전환될 수 있다.
  - 난이도별 시간 prior를 설정할 수 있고, first pass 동안에는 아직 보지 않은 문제를 순차적으로 보도록 환경이 유도된다.
  - second pass 재방문 우선순위에는 recent-entry cooldown과 "이미 실제로 시간을 써본 문제 우선" 정책을 둘 수 있어, 같은 hard problem 사이를 `next`로만 왕복하는 루프를 줄인다.
  - 또한 난이도별 target confidence를 설정해, 대표 시간 안에 어느 정도까지 confidence가 오를지를 보수적으로 조절할 수 있다.
  - 시작 문제 결정 방식은 두 가지 모드가 있다.
    - 기본 모드(`allow_agent_selected_start_problem: false`): reset 시점에 시작 문제가 확정된다. `randomize_start_problem: true`이면 랜덤, false이면 0번 문제로 시작한다.
    - 신규 모드(`allow_agent_selected_start_problem: true`): reset 이후 `current_problem_idx = -1`(sentinel). 에이전트의 첫 번째 action이 시작 문제를 선택하며 switch 시간 비용이 없다. PPO/DQN/heuristic 모두 이 모드를 지원한다.
  - 여러 시험 JSON이 주어지면 reset 시 그중 하나를 샘플링해 episode를 구성할 수 있다.

- `env/dynamics.py`
  - 시간 투입에 따라 문항별 confidence가 상승하는 곡선을 정의한다.
  - 주관식은 `answer_confidence`, 객관식은 정답 선택지 confidence를 중심으로 dynamics가 전개된다.
  - logistic 모델: `p = floor + (1−floor) × sigmoid(θ − β·d − γ·a + α·log(1+t/τ))`
    - 학생 능력(θ), 난도 anchor(d), ambiguity(a), 시간 학습률(α, τ) 4개 축으로 분리
    - `d = 1 − correct_rate`가 기본; `correct_rate`가 없으면 `difficulty` fallback
    - `a = choice_entropy(problem)` (정규화 Shannon entropy)
  - 모든 파라미터는 `config["dynamics"]` 블록으로 조절 가능하다.
  - `hardness = 0.55*difficulty + 0.45*error_rate` 구조는 제거됐다.

- `env/reward.py`
  - step reward 기본형은 `delta expected_total_score` 하나다.
    - `r_t = Σ score_i × p_i(t+Δt) − Σ score_i × p_i(t)`
  - 모든 shaping term(solve_more penalty, low_marginal_gain, saturation, streak, first_pass bonus, difficulty_exit, coverage_bonus, terminal completion, concentration)은 `configs/default.yaml`에서 **기본값 0 / off**로 설정된다.
  - shaping이 필요한 ablation 실험에서는 별도 config에서 해당 키를 켠다.
  - `switch_time_sec` 자체가 이동 비용이므로, 기본 설정에서는 별도 이동 penalty가 필요하지 않다.
  - `next` 출발 시점 shaping이 필요하다면 에이전트가 실제로 관측 가능한 confidence 표현을 기준으로 계산한다.
    - 주관식: `answer_confidence`
    - 객관식: 가장 높은 선택지 confidence
  - `solve_more`의 low-marginal-gain 판정은 문항 타입별 confidence 증가량 기준이다.
    - 주관식: `delta(answer_confidence)`
    - 객관식: `delta(confidence_of_correct_choice)`

- `env/state.py`
  - 시험 진행 상태를 표현하는 `ExamState`, 문항별 진행 정보를 담는 `ProblemProgress`, 상태 열거형 `ProblemStatus`를 정의한다.

- `env/problem.py`
  - 시험 문항 메타데이터(`difficulty`, `correct_rate`, `score`, `error_rate`, `problem_type`, `choice_rate` 등)를 로딩한다.
  - `correct_rate`는 JSON에서 로딩되며, dynamics의 난도 anchor로 1순위 사용된다.
  - `choice_entropy()`, `top2_gap()`, `distractor_mass()` helper가 `choice_rate` 기반 ambiguity 지표를 계산한다.

- `env/student.py`
  - 학생 프로필(`StudentProfile`)과 레벨 기반 synthetic student 생성 로직을 제공한다.

### 3. 에이전트(`agents/`)

- `agents/train_rl.py`
  - Stable-Baselines3 기반 PPO, DQN과 preference-based DPO 학습을 담당한다.
  - PPO는 환경의 `MultiDiscrete` action space를 그대로 사용한다.
  - DQN은 `DiscreteActionWrapper`를 통해 `action_type x target_problem_idx` 조합을 단일 이산 행동으로 펼쳐서 사용한다.
  - DPO는 heuristic teacher가 만든 action preference pair를 이용해 별도 PyTorch policy를 학습한다.
  - `training.strategy_constraint`가 설정되면 baseline 실험용 action wrapper를 적용한다.
    - `fixed_order_free_time`: 문제 순서를 1번에서 30번으로 고정하고, 남은 문제의 최소 풀이 시간을 예약한 뒤 현재 문제를 더 풀지 다음 문제로 넘어갈지만 학습한다.
    - `equal_time_free_order`: 각 문제 풀이 시간을 동일 budget으로 고정하고, 다음에 방문할 문제 순서만 학습한다.
  - 학습 중 checkpoint 저장, progress 출력, 최종 모델 저장, 학습 후 평가까지 수행한다.

- `agents/heuristic_agents.py`
  - 규칙 기반 비교 정책이 들어 있다.
  - 현재 등록된 정책: `equal_time`, `index_order`, `easy_first`, `high_score_first`, `score_time_ratio`, `marginal_gain_greedy`
  - `marginal_gain_greedy`는 매 step마다 모든 문제에 대해 `score × Δconf / (action_time + switch_cost)` 를 계산해 가장 높은 행동을 선택하는 1-step lookahead greedy다. 현재 시뮬레이터가 명시적이므로 강한 baseline이 된다.
  - heuristic 목록은 `HEURISTIC_POLICIES` dict에 등록되어 evaluator에서 자동으로 집계된다.

### 4. 분석(`analysis/`)

- `analysis/evaluator.py`
  - heuristic 또는 RL 정책을 여러 episode에서 실행해 공통 평가 지표를 계산한다.
  - 현재 공통 summary에는 `mean_reward`, `mean_score`, `mean_solved_count`, `mean_visited_count`, `mean_coverage_fraction`, `mean_remaining_time_sec`, `mean_steps`, `mean_top1_time_share`, `mean_top2_time_share`가 포함된다.
  - 타입별 진단 지표로 `mean_objective_dominance_rate`, `mean_subjective_confidence`, `mean_subjective_solved_rate`, `mean_objective_solved_rate`도 함께 계산한다.
- `analysis/trajectory_report.py`
  - RL 또는 heuristic 정책의 실제 trajectory를 점검하는 sanity-check 스크립트다.
  - 첫 episode의 `visit_order`, 문제별 `time_spent_sec`, 타입별 confidence detail, solved 여부, objective/subjective breakdown, `top1/top2 time share`를 함께 출력한다.
  - 재방문 관련 지표는 단순 진입 횟수가 아니라, 문제에 다시 돌아온 뒤 실제로 `solve_more`를 수행한 "의미 있는 revisit session" 기준으로 집계한다.
- `analysis/env_validation.py`
  - RL 없이 environment 자체를 검증하는 스크립트다.
  - 대표 문항을 뽑아 `time -> probability`, `time -> expected score`, `marginal gain` 곡선을 그림으로 저장한다.
  - dynamics monotonicity, easy-vs-hard, objective-vs-subjective, ambiguity effect, reward consistency를 사람이 눈으로 빠르게 확인하는 용도다.
- `analysis/plots.py`
  - 결과 시각화를 위한 코드가 위치한다.

### 5. 유틸리티(`utils/`)

- `utils/io.py`: YAML/JSON config 로딩, 결과 저장
- `utils/seed.py`: 실험 재현성을 위한 seed 설정

## 현재 구현된 학습 대상

이 프로젝트에서 RL 에이전트는 "현재 문제에 시간을 더 쓸지"와 "어느 다음 문제로 이동할지"를 학습한다. 즉, 문제 풀이의 세부 답안 생성은 모델링하지 않고, 시간 투자에 따라 해당 문항의 정답 가능성 또는 기대 점수가 증가하는 과정을 근사해서 전략 최적화 문제로 바꾼 구조다.

각 문항은 다음과 같은 정보를 가진다.

- 문항 번호 `pid`
- 난도 레벨 문자열과 정규화 난도값 `difficulty`
- 배점 `score`
- 정답률 `correct_rate` (JSON 로드 시 읽음; dynamics anchor로 1순위 사용)
- 오답률 `error_rate`
- 객관식/주관식 여부 `problem_type`
- 객관식일 경우 선택지 분포 `choice_rate`

학생은 다음 능력치를 가진다.

- 직접 지정되는 능력 logit `theta`

현재 구현에서 confidence 곡선은 `env/dynamics.py`의 logistic 모델로 결정된다.
- `p_i(t|s) = floor + (1-floor) × sigmoid(θ − β·d − γ·a + α·log(1 + t/τ))`
  - floor: 0.2 (객관식) 또는 subjective_floor (주관식, 기본 0.0)
  - θ: 학생 프로필에 직접 저장된 능력 logit
  - d: 난도 anchor = `difficulty`
  - a: choice entropy (ambiguity; 0=명확, 1=최대혼란)
  - α, τ: 시간 학습 속도·스케일
- `difficulty`, `correct_rate` 등 비공개 수치는 환경 내부에만 사용된다. 반면 `difficulty_level`, `score`, `problem_type`, `error_rate`, 타입별 confidence 슬롯은 observation에 포함된다.
- `choice_rate`는 "학생이 어떤 보기를 선택할 확률"로 직접 쓰지 않는다. 대신 choice entropy, top2_gap 등 **문항 ambiguity feature** 계산에만 사용한다.

## 데이터와 실행 방식

현재 리포지토리에는 30문항 형태의 수학 시험 JSON 데이터(`data/25_math_*.json`)와 학생 preset(`data/student_level_presets.json`), 개별 학생 예시(`data/someone.json`)가 있다.

실행 예시는 다음 흐름으로 이해하면 된다.

1. `main.py --mode train`
   - config를 읽고 PPO, DQN, 또는 DPO를 학습한다.
   - 결과는 `runs/<algo>_<timestamp>/` 아래에 checkpoint, tensorboard log, eval 결과로 저장된다.
   - baseline 1은 `configs/baseline_fixed_order_free_time.yaml`, baseline 2는 `configs/baseline_equal_time_free_order.yaml`로 실행한다.
2. `main.py --mode eval --model-path ...`
   - 저장된 RL 모델을 다시 불러와 여러 episode로 평가한다.
3. `main.py --mode heuristic`
   - heuristic 정책들의 평균 성능 표를 생성한다.

## 산출물

학습 실행 시 다음 파일들이 생성된다.

- `checkpoints/`: 주기적 저장 모델
- `logs/`: tensorboard 로그
- `eval/*.json`: 평균 reward, 평균 score, 평균 solved count, 남은 시간 등
- trajectory sanity check는 `analysis/trajectory_report.py`로 별도 생성 가능
- `config_snapshot.yaml`: 실제 학습에 사용된 설정 스냅샷

## 구현상 참고할 점

- 기본 환경은 Gymnasium 인터페이스를 따르지만, 라이브러리가 없을 때 최소 fallback 클래스도 포함한다.
- DQN은 `MultiDiscrete`를 직접 처리하지 못하므로 wrapper가 필수다.
- 환경 기본 데이터 경로 fallback에는 예전 mock 파일명이 남아 있으므로, 실제 실행 시에는 config 또는 CLI 인자로 현재 존재하는 데이터 파일을 명시하는 편이 안전하다.

## 평가 체계 완료 기준

평가/분석 체계가 정상이라고 보려면 아래를 만족해야 한다.

- `python -m unittest tests.test_evaluation_loops`가 통과한다.
- RL 평가와 heuristic 평가가 공통 summary key를 사용한다.
- 고정 `exam`, 고정 `student`, 고정 `seed`에서 같은 결과가 재현된다.
- `analysis/trajectory_report.py`의 trajectory 해석과 summary 지표가 서로 모순되지 않는다.
- 정책 이상 여부는 `mean_score`만이 아니라 `mean_coverage_fraction`, `mean_top1_time_share`, `mean_top2_time_share`까지 함께 보고 판단한다.
- confidence 구조 변경 이후에는 `mean_objective_dominance_rate`, `mean_subjective_confidence`, `mean_subjective_solved_rate`, `mean_objective_solved_rate`도 함께 본다.

## solved 판정 기준

평가에서 사용하는 `solved_count`는 config의 `evaluation.solved` 기준을 따른다.

- 주관식:
  - `answer_confidence >= subjective_conf_threshold`
- 객관식:
  - 정답 선택지가 가장 높은 confidence를 가져야 한다.
  - 그 confidence가 `objective_conf_threshold` 이상이어야 한다.
  - 동시에 정답 선택지 confidence와 2등 선택지 confidence의 차이가 `objective_margin_threshold` 이상이어야 한다.

기본값은 다음과 같다.

- `subjective_conf_threshold = 0.5`
- `objective_conf_threshold = 0.5`
- `objective_margin_threshold = 0.05`
