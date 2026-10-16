from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

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

# LunarLander: 달 착륙선을 두 깃발 사이 착륙 패드에 부드럽게 내리기.
# 행동 4개: 0=무동작, 1=왼쪽 엔진, 2=메인 엔진, 3=오른쪽 엔진
# 보상(shaped): 패드에 가까울수록/속도가 줄수록/수평일수록 +,
#   다리 접지 +10씩, 착륙 +100, 추락 -100, 엔진 사용마다 연료 페널티.
#   "해결" 기준은 평균 +200이지만 tabular로는 무리 — 여기서는
#   "추락(-100~-200)을 벗어나 0 근처(=착륙 시도)까지 오르는 것"이 목표.
#
# 관측 8차원: [x, y, vx, vy, 기체 각도, 각속도, 왼다리 접지, 오른다리 접지]
# 범위는 랜덤 정책 4.7만 스텝 실측(1%~99% 분위수)에 여유를 더해 결정.
# 극단값(공중제비 돌며 추락할 때의 각도 ±3.6, 각속도 ±7.5)은 어차피
# 실패 상태라 정밀할 필요가 없어 경계 구간에 흡수시킨다.
# 다리 접지 2개는 원래 0/1 이진값이라 구간 2개면 정확히 표현된다.
OBS_LOW = [-1.0, -0.3, -1.5, -1.8, -1.5, -2.0, 0.0, 0.0]
OBS_HIGH = [1.0, 1.6, 1.5, 0.5, 1.5, 2.0, 1.0, 1.0]
N_BINS = [8, 8, 8, 8, 8, 8, 2, 2]
# 상태 수 = 8^6 * 2 * 2 = 약 105만 가지.
# CartPole(2만)의 50배 — 이게 tabular의 한계 지점이다.
# 상태 대부분은 몇 번 못 가보고, 일반화(비슷한 상태끼리 지식 공유)가 안 되니
# 성능이 DQN에 한참 못 미친다. 차원의 저주를 몸으로 배우는 환경.


def make_env(render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    return DiscretizeObservation(env, OBS_LOW, OBS_HIGH, N_BINS)


env = make_env(render_mode=None)
env.reset()

# 하이퍼파라미터
# 상태 수가 많아 재방문이 드물다 → 에피소드를 넉넉히, 탐험도 오래 유지.
learning_rate = 0.1
n_episodes = 50_000
start_epsilon = 1.0
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

# 관람: 학습 전 1판 (엔진을 아무렇게나 뿜다가 추락한다)
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
plt.show()
