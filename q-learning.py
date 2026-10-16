from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

from matplotlib import pyplot as plt

# Agent 생성
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
        """Q-Learning agent 초기화.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        # env.action_space.n => 6
        # np.zeros(6) =>[0. 0. 0. 0. 0. 0.] # 0으로 채워진 길이 6짜리 float 배열
        # defaultdict에 "기본값 생성 함수"(lambda) 를 넘겨서, 없는 키에 접근하면 자동으로 np.zeros(6)이 만들어지도록 함
        # self.q_values[상태][액션] = 행동 가치 몇 점
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # 학습률(α), 샘플 하나는 못 믿으므로 α만큼만 반영한다
        self.lr = learning_rate

        # 누적 보상의 경우 앞으로의 상태가 무한하면 무한히 발산할 수 있다. 따라서 감쇠 인자(γ)를 주어 발산을 막는다.
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        # 상태 s에서 ε-greedy(ε 확률로 랜덤 행동, 1-ε 확률로 정책(argmax_a Q₀)을 따름)로 행동 a를 하면, 환경이 보상 r과 다음 상태 s′를 준다.
        # 초기 ε(initial_epsilon) 부터 시작해서 epsilon_decay 만큼 조금씩 줄여서 최종 ε(final_epsilon)까지 줄인다.
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.
        상태 s(obs)에서 ε-greedy(ε 확률로 랜덤 행동, 1-ε 확률로 정책(argmax_a Q₀)을 따름)로 행동 a를 선택한다.

        Returns:
            action: env.action_space.start(0) 이상 env.action_space.n(6) 미만의 int
        """
        # With probability epsilon: explore (random action)
        # ε 확률로 랜덤 행동
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        # 1-ε 확률로 정책(argmax_a Q₀)을 따름
        # q_values[상태] 리스트 => [0. 0. 0. 0. 0. 0.] => 0번 부터 5번 액션의 행동 가치 점수
        else:
            return int(np.argmax(self.q_values[obs]))

    # 겪은 (s, a, r, s′)로 행동 가치를 갱신한다.
    # 지금 장부에 적힌 값(Q(s,a)) = 지금 장부에 적힌 값(Q(s,a)) + 학습률(α) * 오차
    # Q-learning 오차 = 타겟(앞으로 완벽하게 다닌다면, 이 행동은 얼마짜리인가 Q) - 지금 장부에 적힌 값(Q(s,a))
    # 타겟(앞으로 완벽하게 다닌다면, 이 행동은 얼마짜리인가 Q) = 보상(r) + γ * 다음 상태에서 가능한 최선의 가치(max_a′ Q(s′,a′))
    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        # 다음 상태에서 가능한 최선의 가치(max_a′ Q(s′,a′))
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        # 타겟(앞으로 완벽하게 다닌다면, 이 행동은 얼마짜리인가 Q) = 보상(r) + γ * 다음 상태에서 가능한 최선의 가치(max_a′ Q(s′,a′))
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        # Q-learning 오차 = 타겟(앞으로 완벽하게 다닌다면, 이 행동은 얼마짜리인가 Q) - 지금 장부에 적힌 값(Q(s,a))
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        # # 지금 장부에 적힌 값(Q(s,a)) = 지금 장부에 적힌 값(Q(s,a)) + 학습률(α) * 오차
        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



# 환경 생성
# R, G, Y, B (Colored Locations): 승객이 대기하거나 내릴 수 있는 4개의 공식 승강장
# 상황: 5 * 5 격자판에서 택시가 승객을 태워 지정된 장소에 내려줘야 합니다.
# 상태 공간: 택시 위치(25) * 승객 위치(5) * 목적지 위치(4) = 총 500가지 상태가 존재합니다.
# 행동: 이동(상하좌우), 승객 태우기(Pick-up), 내리기(Drop-off).
# 보상: 목적지에 잘 내리면 +20, 엉뚱한 곳에 내리려 하면 -10, 매 스텝마다 -1.
ENV_ID = "Taxi-v4" # Taxi-v4, FrozenLake-v1, CliffWalking-v1, Blackjack-v1 # 상태가 정수 혹은 tuple인 것들만 가능, float lit / 이미지 등은 불가
env = gym.make(ENV_ID, render_mode="None") # 시각화를 하지 않음 (기본값). 학습 속도가 가장 빠름

# 환경 초기화
env.reset()

# 에이전트 생성
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# 에피소드별 누적 보상(return_queue)과 에피소드별 step 호출 수(액션 수, length_queue)를 기록하는 래퍼
# - 래퍼는 기본 코드를 직접 수정하지 않고도 기존 환경을 편리하게 변경할 수 있는 방법입니다
# - 래퍼는 환경과의 상호 작용 방식을 바꾸는 필터나 수정자와 같다고 생각하면 됩니다.
# - 래퍼를 사용하면 반복적인 코드 작성을 피하고 환경을 더욱 모듈화할 수 있습니다.
# - 또한 여러 래퍼를 연결하여 효과를 조합할 수도 있습니다.
# return_queue
#   - env.reset()으로 에피소드가 시작되면 합계가 0에서 시작하고, 매 스텝 받은 보상이 그대로 더해집니다.
#   - 에피소드가 끝나면(terminated 또는 truncated) 그 합계가 return_queue에 하나 추가됩니다.
#   - 감쇠 인자(γ)는 적용되지 않은 단순 합입니다. γ가 붙은 할인 누적 보상은 Q-value 학습에서 쓰는 개념이고, 이 래퍼는 평가용으로 생(生) 보상을 그냥 더합니다.
# length_queue
#   - 에피소드가 시작(env.reset())되면 카운터가 0이 되고, 행동 하나를 실행할 때마다(env.step(action)) 1씩 증가합니다.
#   - 에피소드가 terminated(승객을 목적지에 내림) 또는 truncated(TimeLimit 초과, Taxi는 기본 200스텝)로 끝나면 그 시점의 카운트가 length_queue에 하나 추가됩니다.
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QLearningAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# 관람: 학습 전 에이전트 1판 플레이
# 관람: 학습 끝난 Q-table로 5판 플레이
eval_env = gym.make(ENV_ID, render_mode="human")
agent.epsilon = 0.0  # 탐험 끄기 q 테이블이 비어 있어서 한 방향으로만 감
for _ in tqdm(range(1)):
    obs, info = eval_env.reset()
    done = False
    for _ in range(50): # 50 step만 돌게함
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated or truncated:
            break
eval_env.close()
agent.epsilon = start_epsilon  # 탐험 재설정

# 학습 환경 플레이 횟수
n_episodes = 50000

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

# 관람: 학습 끝난 Q-table로 5판 플레이
eval_env = gym.make(ENV_ID, render_mode="human")
agent.epsilon = 0.0  # 탐험 끄고 배운 정책만 사용
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
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()