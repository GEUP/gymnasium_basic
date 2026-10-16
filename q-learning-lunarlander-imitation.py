# q-learning-lunarlander.py + 모방학습(imitation learning) 워밍업.
#
# 구조: ① 사람이 직접 조종해 시연(demonstration) 수집
#       ② 시연 transition을 반복 재생해 Q-테이블 사전학습 (DQfD의 pretrain 단계)
#       ③ 그 위에 평소처럼 Q-러닝
#
# 왜 되는가: Q-러닝은 off-policy — transition을 누가 만들었든(사람이든)
# 같은 update 식으로 학습된다. 시연은 "보상이 나오는 영역"에서 출발하게 해줘서
# 무작위 탐험이 착륙 근처를 우연히 발견하길 기다리는 비용을 건너뛴다.
#
# 한계도 같이 보인다: 105만 상태 중 시연 10판이 덮는 건 수백~수천 상태뿐이라,
# 시연 밖 상태에 빠지는 순간 정책이 무너진다(compounding error).
# "모방으로 시작, RL로 초월"이 왜 한 세트인지 몸으로 배우는 스크립트.

import os
import subprocess
import sys
from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

import matplotlib

# macOS에서 pygame(SDL)과 Tk/macosx 계열 GUI 백엔드는 한 프로세스에 못 산다 —
# 학습 전후 '관람'(human 렌더)이 pygame을 쓰므로 그래프 창을 Tk로 열면 크래시.
# 단, PyCharm SciView 같은 IDE 백엔드는 Tk가 아니라 안전하니 그대로 쓴다.
# 위험한 백엔드일 때만 창 없는 agg로 후퇴하고 PNG를 OS 뷰어로 연다.
HEADLESS_PLOT = any(b in matplotlib.get_backend().lower() for b in ("tk", "macosx"))
if HEADLESS_PLOT:
    matplotlib.use("agg")
from matplotlib import pyplot as plt


# 구간화(binning) 래퍼 — q-learning-binning.py와 동일
class DiscretizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, low, high, n_bins):
        super().__init__(env)
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)
        self.n_bins = np.array(n_bins, dtype=np.int64)

    def observation(self, obs):
        clipped = np.clip(obs, self.low, self.high)
        ratio = (clipped - self.low) / (self.high - self.low)
        idx = np.minimum((ratio * self.n_bins).astype(np.int64), self.n_bins - 1)
        return tuple(idx)


class QLearningAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.99,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


ENV_ID = "LunarLander-v3"

# 관측 범위·구간 설계는 q-learning-lunarlander.py 참고 (동일)
OBS_LOW = [-1.0, -0.3, -1.5, -1.8, -1.5, -2.0, 0.0, 0.0]
OBS_HIGH = [1.0, 1.6, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0]
N_BINS = [8, 8, 8, 8, 8, 8, 2, 2]


def make_env(render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    return DiscretizeObservation(env, OBS_LOW, OBS_HIGH, N_BINS)


def discretize_obs(obs):
    # DiscretizeObservation.observation()과 같은 규칙.
    # 시연은 원시(연속) 관측으로 저장하므로, 재생할 때 여기서 이산화한다.
    low, high, n_bins = np.array(OBS_LOW), np.array(OBS_HIGH), np.array(N_BINS)
    clipped = np.clip(obs, low, high)
    ratio = (clipped - low) / (high - low)
    return tuple(np.minimum((ratio * n_bins).astype(np.int64), n_bins - 1))


# ── 모방학습 1단계: 사람 시연 수집 ──
# demo 파일이 없으면 조작 창이 뜬다. 다시 녹화하려면 파일을 지우고 실행.
# 원시 관측으로 저장하므로 나중에 N_BINS를 바꿔도 시연은 재사용된다.
DEMO_PATH = "lunarlander-demos.npz"
N_DEMO_EPISODES = 100
DEMO_FPS = 20  # 기본 50은 사람에게 너무 빠르다. 물리는 그대로, 실시간 속도만 느려진다.


class AlternatingAction(gym.Wrapper):
    """조작 편의용 가상 행동: 여러 엔진을 스텝마다 순환 분사.
    이산 행동은 스텝당 하나뿐이라 동시 분사가 불가능하니, 키를 겹쳐 누르면
    해당 엔진들을 한 스텝씩 돌아가며 분사한다 — 각 엔진을 1/n 추력으로
    동시 분사하는 것의 근사(PWM 원리). env에는 항상 실제 행동(0~3)만 전달되고,
    무엇이 실행됐는지는 info["real_action"]으로 알려준다."""

    CYCLES = {
        4: (1, 2),      # 오른쪽 엔진 ↔ 메인
        5: (3, 2),      # 왼쪽 엔진 ↔ 메인
        6: (3, 1),      # 왼쪽 ↔ 오른쪽
        7: (3, 2, 1),   # 왼쪽 → 메인 → 오른쪽 순환
    }

    def __init__(self, env):
        super().__init__(env)
        self._tick = 0
        self._last_virtual = None
        # play()가 매핑된 모든 행동이 action_space 안에 있는지 검사(assert)하므로
        # 가상 행동까지 포함하도록 겉보기 행동 공간을 넓힌다. (env에는 0~3만 전달됨)
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, **kwargs):
        self._tick = 0
        self._last_virtual = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if action in self.CYCLES:
            if action != self._last_virtual:  # 조합이 바뀌면 순환을 처음부터
                self._tick = 0
                self._last_virtual = action
            cycle = self.CYCLES[action]
            action = cycle[self._tick % len(cycle)]
            self._tick += 1
        else:
            self._last_virtual = None
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["real_action"] = action
        return obs, reward, terminated, truncated, info


def collect_demos(path, n_episodes):
    """사람이 직접 조작해 시연을 모은다. A=왼쪽 엔진, S=메인 엔진, D=오른쪽 엔진, 무입력=무동작.
    좌우 엔진은 기체를 회전시켜서 방향이 직관과 반대로 느껴질 수 있다 — 첫 한두 판은 연습.
    완벽할 필요 없다. '추락보다 나은' 시연이면 Q-테이블 초기화 가치가 있다.
    n_episodes판을 마치면 자동 종료·저장. (ESC로 조기 종료해도 수집분은 저장)"""

    # pygame은 파이썬 게임 라이브러리예요. 창 띄우기, 그림 그리기, 키보드/마우스 입력 받기를 해줍니다.
    import pygame
    # play는 gymnasium이 제공하는 "사람이 env를 직접 플레이하게 해주는" 유틸리티 함수입니다.
    from gymnasium.utils.play import play

    # 기록장 : 게임의 매 스텝마다 각 리스트에 값이 하나씩 추가됩니다. *(Q-learning의 agent.update()가 먹는 재료와 정확히 같은 5개 + 에피소드 번호(episode))
    buf = {"obs": [], "action": [], "reward": [], "terminated": [], "next_obs": [], "episode": []}
    returns = [0.0]

    # on_step — play가 매 스텝 불러주는 콜백
    # 매 스텝마다의 상태t, 상태t+1, 액션, 보상, 종료 여부 를 기록장에 저
    def on_step(obs_t, obs_tp1, action, reward, terminated, truncated, info):
        # play()의 허점: 에피소드 첫 스텝의 obs_t는 reset() 반환값 (관측, info) 튜플이
        # 그대로 넘어온다 (play.py의 obs = env.reset(...)가 언패킹을 안 함). 관측만 꺼낸다.
        if isinstance(obs_t, tuple):
            obs_t = obs_t[0]
        buf["obs"].append(obs_t)
        # play가 넘겨주는 action은 가상 행동(4/5)일 수 있다 — 실제 실행된 행동을 기록
        buf["action"].append(info["real_action"])
        buf["reward"].append(reward)
        # truncated(시간 초과)는 부트스트랩을 끊으면 안 되므로 terminated만 저장
        buf["terminated"].append(terminated)
        buf["next_obs"].append(obs_tp1)
        buf["episode"].append(len(returns) - 1)  # 에피소드 id — 나중에 리턴 기준 선별용
        returns[-1] += reward

        # 하나의 에피소드가 끝끈나면 점수를 출력하고 returns.append(0.0)으로 다음 점수판을 만듬
        # 정해진 길이의 에피소드가 모두 끝나면 종료
        #    pygame은 이벤트 큐 방식으로 동작합니다. 키 누름, 마우스 클릭, 창 닫기 버튼 클릭 같은 사건이 전부 큐에 쌓이고, 프로그램이 하나씩 꺼내 처리해요.
        #    pygame.QUIT은 "사용자가 창의 X 버튼을 눌렀다"는 이벤트인데, event.post()로 그 이벤트를 우리가 큐에 넣을 수 있습니다.
        #    play는 매 프레임 큐를 확인하다가 QUIT을 보면 루프를 끝내죠.
        #    즉 이 한 줄은 "N판 다 모았으니 창 닫기 버튼이 눌린 척해서 play를 종료시킨다"입니다. play에는 "N판 후 종료" 같은 옵션이 없어서 이 우회로를 쓴 거예요.

        if terminated or truncated:
            print(f"시연 {len(returns)}/{n_episodes}판: 리턴 {returns[-1]:.1f}")
            if len(returns) >= n_episodes:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            returns.append(0.0)

    play(
        AlternatingAction(gym.make(ENV_ID, render_mode="rgb_array")),
        # 행동은 한 스텝에 하나뿐(이산 4개)이라 진짜 동시 분사는 불가능하다.
        # 대신 두 키 동시 입력은 가상 행동 4/5로 보내고, AlternatingAction 래퍼가
        # 메인↔사이드를 한 스텝씩 교대로 분사한다.
        # play는 "눌린 키 조합 전체"를 여기서 찾고 없으면 noop 처리하므로,
        # 조합 자체도 반드시 매핑에 들어 있어야 한다.
        keys_to_action={
            "a": 3, "s": 2, "d": 1,
            ("a", "s"): 5, ("s", "d"): 4,
            ("a", "d"): 6, ("a", "s", "d"): 7,
        },
        noop=0,
        zoom=1.5,
        fps=DEMO_FPS,
        callback=on_step,
    )
    n_new = len(returns) - 1  # 마지막 원소는 미완(ESC 중단) 에피소드의 부분 합
    existing = np.load(path) if os.path.exists(path) else None
    if n_new == 0:
        if existing is None:
            raise SystemExit("완주한 시연이 없다. 다시 실행해서 최소 1판은 끝까지 조작할 것.")
        print(f"추가 시연 없음 — 기존 {len(existing['episode_return'])}판 그대로 사용")
        return
    # 미완 에피소드의 transition은 리턴을 모르므로 버린다
    complete = np.array(buf["episode"]) < n_new
    arrays = {k: np.array(v)[complete] for k, v in buf.items()}
    arrays["episode_return"] = np.array(returns[:n_new])
    if existing is not None:
        # 누적: 새 판들의 에피소드 id를 기존 판 수만큼 밀어서 뒤에 이어 붙인다
        arrays["episode"] += len(existing["episode_return"])
        arrays = {k: np.concatenate([existing[k], v]) for k, v in arrays.items()}
    np.savez(path, **arrays)
    print(f"시연 저장: +{n_new}판 (총 {len(arrays['episode_return'])}판, "
          f"{len(arrays['action'])} transition) → {path}")


# 자식 프로세스 모드: 시연만 수집하고 즉시 종료 (아래 subprocess.run이 이 모드로 재실행)
if "--collect-demos" in sys.argv:
    collect_demos(DEMO_PATH, N_DEMO_EPISODES)
    sys.exit(0)

# 시연은 실행할 때마다 "누적"된다 — 새로 플레이한 판들이 기존 파일 뒤에 붙는다.
# 이번 실행에서 추가 수집 없이 기존 시연만 쓰려면 창이 뜨자마자 ESC를 누를 것.
# 처음부터 다시 모으려면 파일을 지우고 실행.
#
# 수집을 자식 프로세스로 격리하는 이유: macOS에서 pygame(SDL)이 앱 객체를
# 바꿔놓아서, 같은 프로세스에서 나중에 matplotlib(Tk) 창을 열면 크래시가 난다.
# 자식만 pygame에 오염되고, 이 프로세스는 깨끗한 채로 그래프를 열 수 있다.
subprocess.run([sys.executable, __file__, "--collect-demos"], check=True)


env = make_env(render_mode=None)
env.reset()

# 하이퍼파라미터 — q-learning-lunarlander.py와 동일
# 시연을 더 신뢰하려면 start_epsilon을 0.5 정도로 낮춰 비교해볼 것 —
# 사전학습된 행동을 일찍 활용하는 대신 탐험 커버리지를 잃는 트레이드오프.
learning_rate = 0.1
n_episodes = 10_000
start_epsilon = 0.3
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.05

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=0.99,
)

# ── 모방학습 2단계: 시연 재생으로 Q-테이블 사전학습 ──
# 한 번 훑어서는 착륙 보상이 궤적 앞쪽 상태까지 전파되지 못하므로 여러 번 반복 재생한다.
# 효과: 시연이 지나간 상태에서는 처음부터 사람의 행동이 greedy가 된다.
# 사전학습의 TD 오차도 training_error에 쌓여 마지막 그래프 맨 앞에 나타난다.
#
# 시연 선별: 리턴 상위 에피소드만 재생한다.
# 행동 복제(BC)라면 실패 시연을 흉내 내는 게 곧 실패 학습이라 선별이 필수지만,
# Q-러닝 재생은 실패도 "이 행동은 추락"이라는 회피 신호로 쓰인다.
# 그래도 선별하면 좋은 궤적에 학습이 집중된다 — 품질 vs 커버리지 트레이드오프.
# 1.0(전부 사용)과 0.5를 바꿔가며 학습 곡선을 비교해볼 것.
KEEP_TOP_FRACTION = 0.1
N_REPLAY_PASSES = 100

demo = np.load(DEMO_PATH)
ep_returns = demo["episode_return"]
n_keep = max(1, int(np.ceil(len(ep_returns) * KEEP_TOP_FRACTION)))
kept_episodes = np.argsort(ep_returns)[::-1][:n_keep]
mask = np.isin(demo["episode"], kept_episodes)
print(f"시연 선별: {len(ep_returns)}판 중 리턴 상위 {n_keep}판 사용 "
      f"(경계 리턴 {ep_returns[kept_episodes].min():.1f}), "
      f"transition {int(mask.sum())}/{len(mask)}")
for ep in kept_episodes:  # 리턴 내림차순 — 무엇을 몇 번 모방학습하는지
    print(f"  에피소드 {ep}: 리턴 {ep_returns[ep]:+.1f}, "
          f"{int((demo['episode'] == ep).sum())} 스텝 × {N_REPLAY_PASSES}회 재생")
demo_transitions = list(zip(
    [discretize_obs(o) for o in demo["obs"][mask]],
    demo["action"][mask].astype(int),
    demo["reward"][mask].astype(float),
    demo["terminated"][mask].astype(bool),
    [discretize_obs(o) for o in demo["next_obs"][mask]],
))
for _ in tqdm(range(N_REPLAY_PASSES), desc="모방 사전학습"):
    for d_obs, d_action, d_reward, d_terminated, d_next_obs in demo_transitions:
        agent.update(d_obs, d_action, d_reward, d_terminated, d_next_obs)

# 관람: RL 학습 전 1판 — 모방 사전학습만 된 상태.
# 시연이 지나간 상태에서는 사람을 흉내 내지만, 못 가본 상태(Q가 전부 0)에서는
# argmax가 무동작(0)이라 그냥 떨어진다. 모방의 '커버리지 한계'를 눈으로 보는 판.
eval_env = make_env(render_mode="human")
agent.epsilon = 0.0
for _ in tqdm(range(1)):
    obs, info = eval_env.reset()
    for _ in range(500):
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            break
eval_env.close()
agent.epsilon = start_epsilon

# 학습
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

# 관람: 학습 후 5판 (속도를 줄이며 내려오면 성공적 — 완벽 착륙까지는 기대하지 말 것)
eval_env = make_env(render_mode="human")
agent.epsilon = 0.0
for _ in tqdm(range(5)):
    obs, info = eval_env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
eval_env.close()

# 학습 결과 분석
def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
PLOT_PATH = "lunarlander-imitation-training.png"
plt.savefig(PLOT_PATH, dpi=150)  # 어느 경로든 파일로는 항상 남긴다
print(f"그래프 저장: {PLOT_PATH}")
if HEADLESS_PLOT:
    subprocess.run(["open", PLOT_PATH])  # Tk 창 대신 macOS 기본 이미지 뷰어로
else:
    plt.show()  # IDE(SciView 등) 백엔드는 플롯 패널에 표시
