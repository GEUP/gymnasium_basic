import gymnasium as gym

# 환경 생성
env = gym.make('CartPole-v1', render_mode="human")

# 환경 초기화
env.reset()

# 무작위 환경 진행
while True:
    # env.render() # 환경의 현재 상태 시각화하기, 환경의 render_mode="human" 이므로 Gymnasium은 우리가 env.render()를 직접 호출하지 않아도 env.step()이 실행될 때마다 자동으로 화면을 업데이트하도록 권장
    action = env.action_space.sample() # 무작위 행동 선택
    observation, reward, terminated, truncated, info = env.step(action) # 환경에서 행동 실행, 전이확률에 따라 다음 상태와 보상, 완료 및 종료 여부를 반환
    if terminated or truncated:
        break

# 환경 종료
env.close()