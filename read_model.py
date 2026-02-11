from stable_baselines3 import PPO, A2C, DDPG, SAC

# 117 종목의 PPO 모델을 불러올 때
model_path = r'C:\Stock_AI\models_1min_test\ppo_0120G0.zip'
model = PPO.load(model_path)

# 모델 정보 확인
print(f"알고리즘: {model.__class__.__name__}")
print(f"학습률(LR): {model.learning_rate}")
print(f"정책망 구조: {model.policy}")