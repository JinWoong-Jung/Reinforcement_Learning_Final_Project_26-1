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
  - 행동은 `MultiDiscrete([2, 문항 수])`이며, 각 step에서
    - `action_type=0`: 현재 문항에 시간을 더 쓰기 (`solve_more`)
    - `action_type=1`: 다른 문항으로 이동하기 (`next`, 재방문 가능)
    를 의미한다.
  - `next`에도 별도의 이동 시간 비용이 들어가며, 현재 문제를 target으로 고르면 자동으로 다른 문제로 redirect된다.
  - 시작 문제는 config에 따라 episode마다 랜덤하게 정해질 수 있다.
  - 여러 시험 JSON이 주어지면 reset 시 그중 하나를 샘플링해 episode를 구성할 수 있다.

- `env/dynamics.py`
  - 시간 투입에 따라 문항별 신뢰도(`confidence_score`)가 상승하는 곡선을 정의한다.
  - 학생 능력, 문항 난도, 오답률을 이용해 신뢰도 상승 속도를 계산한다.

- `env/reward.py`
  - step reward는 직전 상태와 다음 상태 사이의 기대 효용 증가량을 기준으로 계산한다.
  - `next`와 `solve_more` 행동에는 여러 shaping term이 붙을 수 있고, 같은 문제에 오래 머무르거나 기대 점수 증가가 거의 없는 `solve_more`에는 추가 penalty가 붙는다.
  - terminal reward는 시간 초과 또는 step limit 도달 시 penalty를 주고, 전체 방문 coverage에 비례한 bonus를 더한다.

- `env/state.py`
  - 시험 진행 상태를 표현하는 `ExamState`, 문항별 진행 정보를 담는 `ProblemProgress`, 상태 열거형 `ProblemStatus`를 정의한다.

- `env/problem.py`
  - 시험 문항 메타데이터(`difficulty`, `score`, `error_rate`, `problem_type` 등)를 로딩한다.

- `env/student.py`
  - 학생 프로필(`StudentProfile`)과 레벨 기반 synthetic student 생성 로직을 제공한다.

### 3. 에이전트(`agents/`)

- `agents/train_rl.py`
  - Stable-Baselines3 기반 PPO, DQN 학습을 담당한다.
  - PPO는 환경의 `MultiDiscrete` action space를 그대로 사용한다.
  - DQN은 `DiscreteActionWrapper`를 통해 `action_type x target_problem_idx` 조합을 단일 이산 행동으로 펼쳐서 사용한다.
  - 학습 중 checkpoint 저장, progress 출력, 최종 모델 저장, 학습 후 평가까지 수행한다.

- `agents/heuristic_agents.py`
  - 규칙 기반 비교 정책이 들어 있다.
  - 요청하신 문서에서는 heuristic 설명을 제외해야 하므로 별도 규칙 문서에서는 다루지 않는다.

### 4. 분석(`analysis/`)

- `analysis/evaluator.py`
  - heuristic 또는 RL 정책을 여러 episode에서 실행해 평균 점수, 평균 reward, 문항별 평균 시간 등을 계산한다.
- `analysis/plots.py`
  - 결과 시각화를 위한 코드가 위치한다.

### 5. 유틸리티(`utils/`)

- `utils/io.py`: YAML/JSON config 로딩, 결과 저장
- `utils/seed.py`: 실험 재현성을 위한 seed 설정

## 현재 구현된 학습 대상

이 프로젝트에서 RL 에이전트는 "현재 문제에 시간을 더 쓸지"와 "어느 다음 문제로 이동할지"를 학습한다. 즉, 문제 풀이의 세부 답안 생성은 모델링하지 않고, 시간 투자에 따라 해당 문항의 정답 가능성 또는 기대 점수가 증가하는 과정을 근사해서 전략 최적화 문제로 바꾼 구조다.

각 문항은 다음과 같은 정보를 가진다.

- 문항 번호 `pid`
- 난도 레벨 문자열과 정규화 난도값
- 배점 `score`
- 오답률 `error_rate`
- 객관식/주관식 여부 `problem_type`
- 객관식일 경우 선택지 분포 `choice_rate`

학생은 다음 능력치를 가진다.

- 전반 실력 `skill_global`
- 속도 `skill_speed`
- 정확도 `skill_accuracy`
- 스트레스 내성 `stress_tolerance`

현재 구현에서 실제 confidence 곡선은 주로 `skill_global`, `skill_speed`, `skill_accuracy`, `stress_tolerance`, 문항 난도, 오답률에 의해 결정된다.
이때 true `difficulty` 같은 비공개 문항 수치는 환경 내부 dynamics 계산에 사용된다. 반면 `difficulty_level`, `score`, `problem_type`, `pid`, `error_rate`는 현재 observation에 포함된다.

## 데이터와 실행 방식

현재 리포지토리에는 30문항 형태의 수학 시험 JSON 데이터(`data/25_math_*.json`)와 학생 preset(`data/student_level_presets.json`), 개별 학생 예시(`data/someone.json`)가 있다.

실행 예시는 다음 흐름으로 이해하면 된다.

1. `main.py --mode train`
   - config를 읽고 PPO 또는 DQN을 학습한다.
   - 결과는 `runs/<algo>_<timestamp>/` 아래에 checkpoint, tensorboard log, eval 결과로 저장된다.
2. `main.py --mode eval --model-path ...`
   - 저장된 RL 모델을 다시 불러와 여러 episode로 평가한다.
3. `main.py --mode heuristic`
   - heuristic 정책들의 평균 성능 표를 생성한다.

## 산출물

학습 실행 시 다음 파일들이 생성된다.

- `checkpoints/`: 주기적 저장 모델
- `logs/`: tensorboard 로그
- `eval/*.json`: 평균 reward, 평균 score, 평균 solved count, 남은 시간 등
- `config_snapshot.yaml`: 실제 학습에 사용된 설정 스냅샷

## 구현상 참고할 점

- 기본 환경은 Gymnasium 인터페이스를 따르지만, 라이브러리가 없을 때 최소 fallback 클래스도 포함한다.
- DQN은 `MultiDiscrete`를 직접 처리하지 못하므로 wrapper가 필수다.
- 환경 기본 데이터 경로 fallback에는 예전 mock 파일명이 남아 있으므로, 실제 실행 시에는 config 또는 CLI 인자로 현재 존재하는 데이터 파일을 명시하는 편이 안전하다.
