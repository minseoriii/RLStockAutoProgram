import os
import pandas as pd
import numpy as np
import optuna
import wandb
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces
from finrl.config import INDICATORS

# 1. 환경 설정 및 W&B 로그인
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# wandb.login() # API 키 필요시 활성화

# [Custom1MinEnv 클래스] 민서의 1분봉 엔진
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
        return self.state, {}

    def _initiate_state(self):
        # [현금, 종가, 주식수, 지표들..., 시간, 분]
        state = [self.initial_amount] + [self.data.close] + [0] + \
                [self.data[tech] for tech in self.tech_indicator_list]
        return np.array(state, dtype=np.float32)

    def _update_state(self):
        # step 함수에서 다음 상태로 넘어갈 때 사용하는 내부 함수
        state = [self.state[0]] + [self.data.close] + [self.state[2]] + \
                [self.data[tech] for tech in self.tech_indicator_list]
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        self.terminal = self.current_step >= len(self.df) - 1
        if self.terminal: return self.state, 0, self.terminal, False, {}
        
        begin_total_asset = self.state[0] + self.state[1] * self.state[2]
        action = actions[0] * self.hmax
        price = self.state[1]
        
        # 매수/매도 로직
        if action > 0:
            available_cash = self.state[0] // (price * (1 + self.buy_cost_pct[0]))
            buy_num = int(min(available_cash, action))
            self.state[0] -= price * buy_num * (1 + self.buy_cost_pct[0])
            self.state[2] += buy_num
        elif action < 0:
            sell_num = min(abs(action), self.state[2])
            self.state[0] += price * sell_num * (1 - self.sell_cost_pct[0])
            self.state[2] -= sell_num
            
        self.current_step += 1
        self.data = self.df.iloc[self.current_step, :]
        self.state = self._update_state()
        
        end_total_asset = self.state[0] + self.state[1] * self.state[2]
        reward = (end_total_asset - begin_total_asset) * self.reward_scaling
        self.asset_memory.append(end_total_asset)
        return self.state, reward, self.terminal, False, {}

# 2. Optuna 최적화 함수
def objective(trial, train_data, env_kwargs):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    env = DummyVecEnv([lambda: Custom1MinEnv(df=train_data, **env_kwargs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0)
    model.learn(total_timesteps=20000)
    return env.get_attr('asset_memory')[0][-1]

# ==========================================
# 메인 루프: 50종목 순차 학습 (스킵 기능 추가!)
# ==========================================
base_dir = r"C:\Stock_AI"
data_dir = os.path.join(base_dir, "data_parquet")
model_save_dir = os.path.join(base_dir, "models_50_stocks")
os.makedirs(model_save_dir, exist_ok=True)

stock_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')][:50]

for file_name in stock_files:
    tic = file_name.replace('.parquet', '')
    model_path = os.path.join(model_save_dir, f"ppo_{tic}.zip")
    norm_path = os.path.join(model_save_dir, f"vec_norm_{tic}.pkl")

    # ★★★ [스킵 기능] 이미 학습된 종목은 건너뜁니다! ★★★
    if os.path.exists(model_path) and os.path.exists(norm_path):
        print(f"⏩ 스킵: {tic} (이미 학습된 모델이 존재합니다)")
        continue
    
    print(f"🌟 종목 학습 시작: {tic}")
    df = pd.read_parquet(os.path.join(data_dir, file_name))
    train_df = df.iloc[:int(len(df)*0.8)].reset_index(drop=True)
    
    env_kwargs = {
        "stock_dim": 1, "hmax": 500, "initial_amount": 10_000_000,
        "buy_cost_pct": [0.0015], "sell_cost_pct": [0.003], "reward_scaling": 1e-4,
        "state_space": 13, "action_space": 1, 
        "tech_indicator_list": INDICATORS + ['hour', 'minute']
    }

    # [STEP 1] Optuna 최적화 (3회만 빠르게 돌려 최적의 LR 찾기)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_df, env_kwargs), n_trials=3)
    best_lr = study.best_params['lr']

    # [STEP 2] W&B 연동 및 본 학습
    wandb.init(project="Project_50_Stocks", name=f"Final_{tic}", reinit=True)
    
    env_train = DummyVecEnv([lambda: Custom1MinEnv(df=train_df, **env_kwargs)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = PPO("MlpPolicy", env_train, learning_rate=best_lr, verbose=1, tensorboard_log=f"./logs/{tic}")
    model.learn(total_timesteps=200000)
    
    # 모델 및 정규화 데이터 저장
    model.save(model_path.replace(".zip", ""))
    env_train.save(norm_path)
    
    wandb.finish()

print("🏁 50종목 전체 학습 완료! 이제 진짜 푹 자러 가자, 민서야.")