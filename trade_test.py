import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, SAC
import gymnasium as gym
from gymnasium import spaces
from finrl.config import INDICATORS

# 라이브러리 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# [Custom1MinEnv 클래스 정의 - 이전과 동일]
class Custom1MinEnv(gym.Env):
    def __init__(self, df, stock_dim, hmax, initial_amount, buy_cost_pct, sell_cost_pct, reward_scaling, state_space, action_space, tech_indicator_list, **kwargs):
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space_dim = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.data = self.df.iloc[self.current_step, :]
        self.state = self._initiate_state()
        self.asset_memory = [self.initial_amount]
        self.terminal = False 
        return self.state, {}

    def _initiate_state(self):
        prices = [self.data.close]
        shares = [0]
        indic = [self.data[tech] for tech in self.tech_indicator_list]
        state = [self.initial_amount] + prices + shares + indic
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.terminal = self.current_step >= len(self.df) - 1
        if self.terminal: return self.state, 0, self.terminal, False, {}
        price = self.data.close
        action = actions[0] * self.hmax
        cash, shares = self.state[0], self.state[2]
        if action > 0:
            buy_num = min(cash // (price * (1 + self.buy_cost_pct[0])), action)
            cash -= price * buy_num * (1 + self.buy_cost_pct[0]); shares += buy_num
        elif action < 0:
            sell_num = min(shares, abs(action))
            cash += price * sell_num * (1 - self.sell_cost_pct[0]); shares -= sell_num
        self.current_step += 1
        self.data = self.df.iloc[self.current_step, :]
        self.state = np.array([cash, price, shares] + [self.data[tech] for tech in self.tech_indicator_list], dtype=np.float32)
        total_asset = cash + shares * self.data.close
        self.asset_memory.append(total_asset)
        return self.state, 0, self.terminal, False, {}

# ==========================================
# 실행 설정
# ==========================================
base_dir = r"C:\Stock_AI"
parquet_path = os.path.join(base_dir, "data_parquet", "0120G0.parquet")
model_dir = os.path.join(base_dir, "models_1min_test")
model_types = {'ppo': PPO, 'a2c': A2C, 'ddpg': DDPG, 'sac': SAC}

df = pd.read_parquet(parquet_path)
test_df = df.iloc[int(len(df)*0.8):].reset_index(drop=True)

# 다른 모델들이 움직이게 하기 위해 수수료 0, hmax 500으로 상향 조정
env_kwargs = {
    "stock_dim": 1, "hmax": 500, "initial_amount": 10_000_000,
    "buy_cost_pct": [0.00], "sell_cost_pct": [0.00], "reward_scaling": 1e-4,
    "state_space": 13, "action_space": 1, 
    "tech_indicator_list": INDICATORS + ['hour', 'minute']
}

all_histories = {}
ensemble_actions = []

print("📊 모델별 수익률 및 행동 분석 시작...")

# 1. 모델별 개별 테스트 (학습 코드 삭제하고 '테스트'만 진행)
for name, model_class in model_types.items():
    model_path = os.path.join(model_dir, f"{name}_0120G0.zip")
    
    if os.path.exists(model_path):
        model = model_class.load(model_path)
        env = Custom1MinEnv(df=test_df, **env_kwargs)
        obs, _ = env.reset()
        done = False
        model_acts = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            model_acts.append(action[0]) # 앙상블을 위해 액션 저장
            obs, _, done, _, _ = env.step(action)
        
        all_histories[name] = env.asset_memory
        ensemble_actions.append(model_acts)
        
        ret = ((env.asset_memory[-1] / 10_000_000) - 1) * 100
        print(f"💰 {name.upper()} 수익률: {ret:.2f}% | 마지막 Action: {model_acts[-1]:.4f}")
    else:
        print(f"❌ {name} 모델 파일을 찾을 수 없습니다.")

# 2. 앙상블 결과 계산
if ensemble_actions:
    avg_actions = np.mean(ensemble_actions, axis=0)
    env = Custom1MinEnv(df=test_df, **env_kwargs)
    obs, _ = env.reset()
    for act in avg_actions:
        obs, _, _, _, _ = env.step([act])
    all_histories['Ensemble'] = env.asset_memory
    print(f"🚀 앙상블 최종 수익률: {((env.asset_memory[-1]/10_000_000)-1)*100:.2f}%")

# 3. 그래프 그리기
plt.figure(figsize=(12, 7))
colors = {'ppo': 'blue', 'a2c': 'green', 'ddpg': 'orange', 'sac': 'red', 'Ensemble': 'black'}
styles = {'Ensemble': '--'}

for name, history in all_histories.items():
    plt.plot(history, label=name.upper(), color=colors.get(name, 'gray'), 
             linestyle=styles.get(name, '-'), linewidth=2 if name == 'Ensemble' else 1.5)

plt.title("Trading Strategy Comparison (Stock: 0120G0)")
plt.xlabel("Time (Minutes)")
plt.ylabel("Total Asset (KRW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(base_dir, "final_report_plot.png"))
print(f"💾 비교 보고서 그래프 저장 완료: final_report_plot.png")