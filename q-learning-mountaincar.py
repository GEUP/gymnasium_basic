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


ENV_ID = "MountainCar-v0"

# MountainCar 관측 2차원: [차 위치, 차 속도]
# 두 차원 모두 공식 범위가 유한이라(CartPole의 속도처럼 inf가 없음) 그대로 쓴다.
#   위치: [-1.2, 0.6]  (0.5 이상이면 깃발 도달 = 성공 종료)
#   속도: [-0.07, 0.07]
# 행동 3개: 0=왼쪽으로 가속, 1=가속 안 함, 2=오른쪽으로 가속
# 보상: 매 스텝 -1 (빨리 도달할수록 이득), 200스텝 안에 못 가면 truncated
#   => 랜덤 정책은 거의 항상 -200. 깃발에 닿아 -1 행렬이 끊기는 경험이
#      우연히 쌓이면서부터 학습이 시작된다.
OBS_LOW = [-1.2, -0.07]
OBS_HIGH = [0.6, 0.07]
N_BINS = [20, 20]  # 상태 수 = 20*20 = 400가지 (2차원이라 CartPole보다 훨씬 작다)


def make_env(render_mode=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    return DiscretizeObservation(env, OBS_LOW, OBS_HIGH, N_BINS)


env = make_env(render_mode=None)
env.reset()

# 하이퍼파라미터
# discount_factor=0.99: 보상이 "깃발 도달로 -1 행렬이 끊김" 하나뿐이고
# 에피소드가 최대 200스텝으로 길어서, γ가 낮으면(0.95) 그 신호가
# 출발 지점까지 전파되기 전에 소멸한다. 지평선을 길게 잡아야 하는 환경.
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

# 관람: 학습 전 1판 (아무것도 몰라서 골짜기에서 부르르 떨기만 한다)
eval_env = make_env(render_mode="human")
agent.epsilon = 0.0
for _ in tqdm(range(1)):
    obs, info = eval_env.reset()
    for _ in range(200):
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

# 관람: 학습 후 5판 (좌우로 흔들며 관성을 모아 올라가면 성공)
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
