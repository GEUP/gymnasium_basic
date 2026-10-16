from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt


# 구간화(binning) 래퍼
# 연속 관측값(float 배열)은 해시가 안 돼서 Q-table(dict)의 키로 쓸 수 없다.
# 각 차원을 n_bins개 구간으로 잘라 "몇 번째 구간에 속하는지"(int)로 바꾸면
# 관측값이 int 튜플이 되어 dict 키로 쓸 수 있다. => tabular Q-learning 가능
#
# 예: CartPole 관측 [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
#     (0.03, -0.4, 0.02, 0.7) --binning--> (5, 3, 5, 7)
#
# 주의: 구간을 잘게 나눌수록 정밀하지만 상태 수가 n_bins^4로 폭발해서 학습이 느려진다.
#       CartPole은 차원당 10~20개면 충분하다.
class DiscretizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, low, high, n_bins):
        """
        Args:
            low: 각 차원의 하한. 실제 관측 공간이 무한대(inf)인 차원은
                 "현실적으로 나올 만한 범위"로 직접 잘라줘야 한다.
            high: 각 차원의 상한
            n_bins: 각 차원을 몇 개 구간으로 나눌지
        """
        super().__init__(env)
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)
        self.n_bins = np.array(n_bins, dtype=np.int64)

    def observation(self, obs):
        # 1) 범위를 벗어난 값은 경계로 자른다 (속도 같은 무한대 차원 대비)
        clipped = np.clip(obs, self.low, self.high)
        # 2) [low, high] 구간에서의 위치를 0.0~1.0 비율로 바꾼다
        ratio = (clipped - self.low) / (self.high - self.low)
        # 3) 비율 * 구간 수 = 구간 번호 (정확히 1.0이면 마지막 구간으로)
        idx = np.minimum((ratio * self.n_bins).astype(np.int64), self.n_bins - 1)
        # 4) 튜플은 해시 가능 => Q-table의 키로 사용 가능
        return tuple(idx)


# Agent 생성 (q-learning.py와 동일 — 관측값이 해시 가능하기만 하면 그대로 동작)
class QLearningAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.env = env

        # Q-table: (상태 튜플) -> 각 액션의 가치 배열
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs) -> int:
        # ε 확률로 랜덤 행동 (탐험)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # 1-ε 확률로 현재까지 배운 최선의 행동 (활용)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        # 다음 상태에서 가능한 최선의 가치 (에피소드가 끝났으면 0)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # 타겟 = 보상 + γ * 다음 상태 최선 가치 (Bellman equation)
        target = reward + self.discount_factor * future_q_value
        # 오차 = 타겟 - 현재 추정치
        temporal_difference = target - self.q_values[obs][action]
        # 오차 방향으로 학습률만큼 이동
        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


ENV_ID = "CartPole-v1"

# CartPole 관측 4차원: [카트 위치, 카트 속도, 막대 각도(rad), 막대 각속도]
# 위치/각도의 공식 범위는 ±4.8, ±0.418이지만 그 전에 에피소드가 종료(±2.4, ±0.2095)되므로
# 실제 나오는 범위 근처로 잡는 게 구간을 알뜰하게 쓴다.
# 속도/각속도는 공식 범위가 ±inf라서 현실적인 값으로 직접 자른다.
OBS_LOW = [-2.4, -3.0, -0.21, -3.5]
OBS_HIGH = [2.4, 3.0, 0.21, 3.5]
N_BINS = [12, 12, 12, 12]  # 상태 수 = 12^4 = 20,736가지 (Taxi 500가지보다 크지만 감당 가능)


def make_env(render_mode=None):
    """학습용/관람용 환경 모두 반드시 같은 binning 래퍼를 씌운다.
    (래퍼가 다르면 학습 때의 상태 키와 관람 때의 상태 키가 달라져 Q-table을 못 찾는다)"""
    env = gym.make(ENV_ID, render_mode=render_mode)
    return DiscretizeObservation(env, OBS_LOW, OBS_HIGH, N_BINS)


# 환경 생성 (render_mode=None: 시각화 없음, 학습 속도 최대)
env = make_env(render_mode=None)
env.reset()

# 하이퍼파라미터
# CartPole은 매 스텝 +1 보상(빽빽한 보상)이라 Taxi보다 학습률을 높게 잡아도 안정적이다.
learning_rate = 0.1
n_episodes = 30_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.05

# 에피소드별 누적 보상/길이 기록 래퍼
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# 관람: 학습 전 에이전트 1판 플레이 (Q-table이 비어 있어 금방 쓰러진다)
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

# 관람: 학습 끝난 Q-table로 5판 플레이 (막대가 오래 버티면 성공)
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
