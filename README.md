# gymnasium_basic

[Gymnasium](https://gymnasium.farama.org/) 환경으로 **테이블형(tabular) Q-learning**을 단계별로 실습하는 예제 모음.
이산 상태 환경에서 시작해, 연속 상태를 구간화(binning)해서 다루는 법, 사람 시연으로 학습을 가속하는 모방학습까지 다룹니다.

## 설치

Python 3.12 이상, [uv](https://docs.astral.sh/uv/) 기준.

```bash
uv venv
uv pip install gymnasium "gymnasium[box2d]" pygame tqdm matplotlib
```

> `gymnasium[box2d]`는 LunarLander에 필요합니다. 빌드 오류가 나면 `brew install swig`를 먼저 실행하세요.

## 스크립트

| 파일 | 환경 | 내용 |
|---|---|---|
| `CortPole.py` | CartPole-v1 | 무작위 행동으로 환경을 띄워 보는 최소 예제 |
| `q-learning.py` | Taxi-v4 | 기본 Q-learning. 상태가 정수인 환경(FrozenLake, CliffWalking, Blackjack 등)으로 바꿔 실행 가능 |
| `q-learning-cartpole.py` | CartPole-v1 | 연속 관측을 구간화해 Q-테이블로 학습 (차원당 12구간, 12⁴ 상태) |
| `q-learning-mountaincar.py` | MountainCar-v0 | 2차원 관측 구간화 (20×20 = 400 상태) |
| `q-learning-acrobot.py` | Acrobot-v1 | cos/sin 쌍을 `atan2`로 각도 복원해 6차원 → 4차원 축소 후 구간화 |
| `q-learning-lunarlander.py` | LunarLander-v3 | 8차원 관측 구간화 |
| `q-learning-lunarlander-imitation.py` | LunarLander-v3 | 사람 시연으로 Q-테이블 사전학습 후 Q-learning (DQfD 방식) |

실행:

```bash
uv run python q-learning-cartpole.py
```

각 스크립트는 학습 → 보상/에피소드 길이/TD 오차 그래프 출력 → 학습된 에이전트 플레이(human 렌더) 순으로 진행됩니다.

## 모방학습 (`q-learning-lunarlander-imitation.py`)

1. **시연 수집** — 실행하면 pygame 창이 뜨고 직접 착륙선을 조종합니다.
   - `A` 왼쪽 엔진, `S` 메인 엔진, `D` 오른쪽 엔진, 무입력 = 무동작
   - 두 키 동시 입력은 메인/사이드 엔진을 한 스텝씩 번갈아 분사
   - `ESC`로 조기 종료해도 완주한 판은 저장됩니다.
2. **사전학습** — 리턴 상위 시연만 골라 transition을 반복 재생해 Q-테이블을 초기화합니다.
3. **Q-learning** — 그 위에서 평소처럼 학습하고, 결과 그래프를 `lunarlander-imitation-training.png`로 저장합니다.

시연은 `lunarlander-demos.npz`에 **누적**됩니다. 추가 수집 없이 기존 시연만 쓰려면 창이 뜨자마자 `ESC`, 처음부터 다시 모으려면 파일을 지우고 실행하세요.

## 데이터 파일

- `lunarlander-demos.npz` — 수집된 LunarLander 시연 데이터
- `lunarlander-imitation-training.png` — 모방학습 실행 결과 그래프
- `demo_test3.npz` — 시연 수집 테스트용 샘플
