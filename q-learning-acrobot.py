from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt


# Acrobot 전용 전처리 래퍼: 6차원 -> 4차원
# 원본 관측: [cosθ1, sinθ1, cosθ2, sinθ2, ω1, ω2]
# cos/sin 쌍은 "각도 하나"의 중복 표현이다 (신경망에는 연속성 때문에 유리하지만
# binning에는 차원만 늘리는 낭비). atan2(sin, cos)로 각도를 복원하면
# [θ1, θ2, ω1, ω2] 4차원이 되어 상태 수가 구간수^6 -> 구간수^4로 줄어든다.
# (주의: θ가 ±π 경계를 넘을 때 첫 구간/마지막 구간으로 갈라지는 단절이 있지만,
#  Q-learning은 상태를 독립적으로 학습하므로 성능에 큰 문제는 없다)
class AcrobotAngleObservation(gym.ObservationWrapper):
    def observation(self, obs):
        cos1, sin1, cos2, sin2, w1, w2 = obs
        theta1 = np.arctan2(sin1, cos1)  # -π ~ +π
        theta2 = np.arctan2(sin2, cos2)
        return np.array([theta1, theta2, w1, w2])


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


ENV_ID = "Acrobot-v1"

# Acrobot: 2관절 로봇 팔. 아래 관절(관절2)에만 토크를 줄 수 있고,
# 흔들어서 팔 끝을 목표 높이(첫 링크 길이만큼 위)까지 올리면 성공.
# 행동 3개: 0=토크 -1, 1=토크 0, 2=토크 +1
# 보상: 매 스텝 -1 (MountainCar처럼 "빨리 도달"이 목표), 500스텝이면 truncated
#
# 전처리 후 관측 4차원: [θ1, θ2, ω1, ω2]
#   θ1, θ2: atan2로 복원한 각도, 범위는 정확히 [-π, π]
#   ω1, ω2: 공식 범위는 ±4π(12.57), ±9π(28.27)지만
#           랜덤 정책 15만 스텝 실측 결과 최대 ±5.4, ±10.1 → 여유 있게 ±6, ±11로 자름
OBS_LOW = [-np.pi, -np.pi, -6.0, -11.0]
OBS_HIGH = [np.pi, np.pi, 6.0, 11.0]
N_BINS = [12, 12, 12, 12]  # 상태 수 = 12^4 = 20,736가지 (원본 6차원이면 12^6 = 300만!)


def make_env(render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = AcrobotAngleObservation(env)  # 6차원 -> 4차원
    return DiscretizeObservation(env, OBS_LOW, OBS_HIGH, N_BINS)


env = make_env(render_mode=None)
env.reset()

# 하이퍼파라미터
# MountainCar와 같은 구조의 보상(-1/스텝, 도달하면 끝)이라 γ=0.99로 지평선을 길게.
learning_rate = 0.1
n_episodes = 20_000
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

# 관람: 학습 전 1판 (아래로 축 늘어져서 힘없이 흔들리기만 한다)
eval_env = make_env(render_mode="human")
agent.epsilon = 0.0
for _ in tqdm(range(1)):
    obs, info = eval_env.reset()
    for _ in range(50):
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

# 관람: 학습 후 5판 (그네 타듯 흔들다가 팔 끝을 목표선 위로 차올리면 성공)
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
