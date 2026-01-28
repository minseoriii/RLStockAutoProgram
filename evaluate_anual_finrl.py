#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# FinRL 라이브러리
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# ★ [수정] 고장 난 라이브러리 환경 대신 우리가 만든 커스텀 환경을 쓸 거라 import 제거함
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv 
from finrl.config import INDICATORS

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ★ [핵심] 수제 엔진 (CustomStockEnv) 정의 
# finrl stocktradingenv 라이브러리 버그 (잔고 무시하고 수십억거래 , HMAX 무시)를 원천 차단함 
# 두가지 로직 이외 설정은 동일 
# ==========================================
import gymnasium as gym
from gymnasium import spaces

class CustomStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, hmax, initial_amount, buy_cost_pct, sell_cost_pct, reward_scaling, state_space, action_space, tech_indicator_list, day=0, initial=True, previous_state=[], model_name='', mode='', iteration='', **kwargs):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False
        self.make_plots = False
        self.print_verbosity = 10
        self.turbulence_threshold = 140
        self.risk_indicator_col = 'turbulence'
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.state = self._initiate_state()
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index+1+self.stock_dim] > 0:
                if self.state[index+1] > 0:
                    sell_num_shares = min(abs(action), self.state[index+1+self.stock_dim])
                    sell_amount = self.state[index+1] * sell_num_shares * (1- self.sell_cost_pct[index])
                    self.state[0] += sell_amount
                    self.state[index+1+self.stock_dim] -= sell_num_shares
                    self.cost += self.state[index+1] * sell_num_shares * self.sell_cost_pct[index]
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares
        return _do_sell_normal()

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index+1] > 0:
                available_amount = self.state[0] // (self.state[index+1] * (1 + self.buy_cost_pct[index]))
                buy_num_shares = min(available_amount, action)
                buy_num_shares = int(buy_num_shares)
                buy_amount = self.state[index+1] * buy_num_shares * (1+ self.buy_cost_pct[index])
                self.state[0] -= buy_amount
                self.state[index+1+self.stock_dim] += buy_num_shares
                self.cost += self.state[index+1] * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares
        return _do_buy()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = actions * self.hmax 
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index: self._sell_stock(index, actions[index])
            for index in buy_index: self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.state = self._update_state()
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling
            return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.state = self._initiate_state()
        self.asset_memory = [self.initial_amount]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        return self.state, {}
    
    def _initiate_state(self):
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                stock_list = self.df.tic.unique()
                amount_list = [0] * len(stock_list)
                prices = self.data.close.values.tolist()
            else:
                amount_list = [0]
                prices = [self.data.close]
            # 가격 먼저, 수량 나중
            state = [self.initial_amount] + prices + amount_list + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        else:
            state = self.previous_state
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            stock_list = self.df.tic.unique()
            prices = self.data.close.values.tolist()
            amount_list = list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
        else:
            amount_list = [self.state[self.stock_dim+1]]
            prices = [self.data.close]
        state = [self.state[0]] + prices + amount_list + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1: date = self.data.date.unique()[0]
        else: date = self.data.date
        return date
    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def calculate_sharpe(df_account):
    df_account['daily_return'] = df_account['account_value'].pct_change(1)
    if df_account['daily_return'].std() == 0: return 0
    sharpe = (252 ** 0.5) * df_account['daily_return'].mean() / df_account['daily_return'].std()
    return sharpe

def analyze_market_benchmark(df_year):
    tickers = df_year['tic'].unique()
    returns = []
    for tic in tickers:
        df_tic = df_year[df_year['tic'] == tic]
        if df_tic.empty: continue
        start_price = df_tic.iloc[0]['close']
        end_price = df_tic.iloc[-1]['close']
        if start_price == 0: continue
        ret = (end_price - start_price) / start_price * 100
        returns.append({'tic': tic, 'return': ret})
    
    df_ret = pd.DataFrame(returns)
    avg_return = df_ret['return'].mean()
    top_5 = df_ret.sort_values('return', ascending=False).head(5)
    
    msg = f"      📊 [시장 기준] 평균 수익률: {avg_return:.2f}% (Buy & Hold)\n"
    msg += f"      🔥 [시장 Top 5] 대박 종목:\n"
    for _, row in top_5.iterrows():
        msg += f"         - {row['tic']}: {row['return']:.2f}%\n"
    return msg

# ==========================================
# 3. 설정
# ==========================================
TRAIN_START_DATE = "2018-01-01" 
TARGET_YEARS = [2022, 2023, 2024] 
MODELS_TO_RUN = ['ppo', 'a2c', 'ddpg', 'sac'] # 개인전 모델
ENSEMBLE_CANDIDATES = ['ppo', 'a2c', 'ddpg'] # 앙상블 후보
TRAIN_TIMESTEPS = 50000 
INITIAL_CAPITAL = 10_000_000 
HMAX_VAL = 5000 

base_dir = r"C:\Stock_AI"
data_dir = os.path.join(base_dir, 'data')
save_model_dir = os.path.join(base_dir, 'rolling_models') 
log_dir = os.path.join(base_dir, 'rolling_logs')          

for d in [save_model_dir, log_dir]:
    if not os.path.exists(d): os.makedirs(d)

print(f"🥊 [System] 통합 챌린지 (Skip 기능 + 앙상블 포함) 시작")

# ==========================================
# 4. 데이터 로드
# ==========================================
data_path = os.path.join(data_dir, 'krx_top200_data.csv')
if not os.path.exists(data_path):
    print("❌ 데이터 파일 없음!")
    exit()

df = pd.read_csv(data_path)
df.columns = df.columns.str.lower()
if 'tic_name' in df.columns: df = df.drop(columns=['tic_name'])

fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=False, use_turbulence=False, user_defined_feature=False)
processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(pd.MultiIndex.from_product([list_date, list_ticker], names=["date", "tic"]))
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])
processed_full = processed_full.fillna(0)

final_summary = []

# ==========================================
# 5. 메인 루프
# ==========================================
for target_year in TARGET_YEARS:
    print(f"\n" + "="*60)
    print(f"🚀 [SEASON {target_year}] 리그 시작")
    print("="*60)
    
    # 공통 데이터
    test_start = f"{target_year}-01-01"
    test_end = f"{target_year}-12-31"
    test_data = data_split(processed_full, test_start, test_end)
    
    if test_data.empty: continue

    # 시장 분석 출력
    print(analyze_market_benchmark(test_data).strip())
    print("-" * 40)

    price_matrix = test_data.pivot(index='date', columns='tic', values='close')
    unique_dates = test_data['date'].unique()
    list_ticker_test = test_data.tic.unique().tolist()

    stock_dim = len(test_data.tic.unique())
    state_space = 1 + 2*stock_dim + len(INDICATORS)*stock_dim
    
    env_kwargs = {
        "hmax": HMAX_VAL, "initial_amount": INITIAL_CAPITAL,
        "num_stock_shares": [0]*stock_dim, "buy_cost_pct": [0.0015]*stock_dim,
        "sell_cost_pct": [0.003]*stock_dim, "state_space": state_space,
        "stock_dim": stock_dim, "tech_indicator_list": INDICATORS,
        "action_space": stock_dim, "reward_scaling": 1e-4
    }

    # -----------------------------------------------------
    # [PART 1] 개인전 (Individual Models) + SKIP 기능
    # -----------------------------------------------------
    train_end_date = f"{target_year-1}-12-31" 
    train_data = data_split(processed_full, TRAIN_START_DATE, train_end_date)
    e_train_gym = CustomStockEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    for algo in MODELS_TO_RUN:
        model_name = f"{algo}_{target_year}" 
        log_name = f"log_{target_year}_{algo.upper()}.csv"
        log_path = os.path.join(log_dir, log_name)
        model_path = os.path.join(save_model_dir, model_name)

        print(f"   👤 [{algo.upper()}] 확인 중...", end=" ")

        # ★ [핵심] 파일 있으면 스킵!
        if os.path.exists(log_path):
            print(f"⏩ 이미 완료됨! (로그 로드)")
            # 결과 요약을 위해 로그 파일 읽어서 수익률 계산
            try:
                existing_log = pd.read_csv(log_path)
                final_cash_row = existing_log.iloc[-1]
                # 자산 계산이 복잡하므로 간단히 표시하거나, 로그에 자산 컬럼이 있다면 사용
                # 여기서는 스킵 메시지만 띄우고 넘어가되, 모델 파일은 있어야 앙상블 가능
            except:
                pass
            continue
        
        # 파일 없으면 학습 진행
        print(f"🔥 학습 시작!")
        try:
            if algo == 'ppo': model = PPO('MlpPolicy', env_train, verbose=1)
            elif algo == 'a2c': model = A2C('MlpPolicy', env_train, verbose=1)
            elif algo == 'ddpg': model = DDPG('MlpPolicy', env_train, verbose=1)
            elif algo == 'sac': model = SAC('MlpPolicy', env_train, verbose=1)
            
            model.learn(total_timesteps=TRAIN_TIMESTEPS)
            model.save(model_path)
            
            # 테스트
            env_test = CustomStockEnv(df=test_data, **env_kwargs)
            obs, _ = env_test.reset()
            done = False
            trade_history = []
            prev_holdings = np.array([0] * stock_dim)
            curr_step = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_res = env_test.step(action)
                if len(step_res) == 5: obs, _, term, trunc, _ = step_res; done = term or trunc
                else: obs, _, done, _ = step_res
                
                curr_state = env_test.state
                curr_cash = curr_state[0]
                curr_holdings = np.array(curr_state[1+stock_dim : 1+2*stock_dim]) # 인덱스 주의 (가격, 수량 순서 수정됨)
                diff = curr_holdings - prev_holdings
                date_str = unique_dates[curr_step] if curr_step < len(unique_dates) else "End"
                
                for i, change in enumerate(diff):
                    if abs(change) > 0:
                        ticker = list_ticker_test[i]
                        try: real_price = price_matrix.loc[date_str, ticker]
                        except: real_price = 0
                        if real_price > 0:
                            trade_history.append({
                                "Date": date_str, "Ticker": ticker, "Type": "매수" if change > 0 else "매도",
                                "Price": real_price, "Shares": abs(change),
                                "Cash": curr_cash, "Total_Value": abs(change) * real_price
                            })
                prev_holdings = curr_holdings
                curr_step += 1
            
            final_val = env_test.asset_memory[-1]
            ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            print(f"      👉 결과: {ret:.2f}%")
            
            final_summary.append({'Year': target_year, 'Model': algo.upper(), 'Return': ret})
            if trade_history:
                pd.DataFrame(trade_history).to_csv(log_path, index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"      ❌ 에러 발생: {e}")

    # -----------------------------------------------------
    # [PART 2] 앙상블 (Ensemble)
    # -----------------------------------------------------
    ens_log_name = f"log_{target_year}_ENSEMBLE.csv"
    ens_log_path = os.path.join(log_dir, ens_log_name)

    print(f"   🛡️ [앙상블] 준비 중...", end=" ")
    
    if os.path.exists(ens_log_path):
        print(f"⏩ 이미 완료됨!")
        continue

    print("선발전 시작!")
    # 검증용 데이터 (작년 10~12월)
    ens_val_start = f"{target_year-1}-10-01"
    ens_val_end = f"{target_year-1}-12-31"
    ens_val_data = data_split(processed_full, ens_val_start, ens_val_end)
    
    best_sharpe = -999
    best_model_obj = None
    best_algo_name = "CASH"

    for algo in ENSEMBLE_CANDIDATES:
        model_name = f"{algo}_{target_year}"
        model_path = os.path.join(save_model_dir, model_name + ".zip") # zip 확장자 주의

        if not os.path.exists(model_path):
            continue # 모델 파일 없으면 패스

        try:
            if algo == 'ppo': model = PPO.load(model_path)
            elif algo == 'a2c': model = A2C.load(model_path)
            elif algo == 'ddpg': model = DDPG.load(model_path)
            
            # 검증
            e_val = CustomStockEnv(df=ens_val_data, **env_kwargs)
            obs, _ = e_val.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_res = e_val.step(action)
                if len(step_res) == 5: obs, _, term, trunc, _ = step_res; done = term or trunc
                else: obs, _, done, _ = step_res
            
            df_val = pd.DataFrame(e_val.asset_memory)
            df_val.columns = ['account_value']
            sharpe = calculate_sharpe(df_val)
            
            # 하락장 방어 로직: 샤프지수 0 이상이어야 선발
            if sharpe > best_sharpe and sharpe > 0:
                best_sharpe = sharpe
                best_model_obj = model
                best_algo_name = algo.upper()
        except:
            continue

    print(f"      👑 선발 모델: {best_algo_name} (검증 Sharpe: {best_sharpe:.2f})")

    # 실전 투입
    env_test = CustomStockEnv(df=test_data, **env_kwargs)
    obs, _ = env_test.reset()
    done = False
    trade_history = []
    prev_holdings = np.array([0] * stock_dim)
    curr_step = 0
    
    while not done:
        if best_model_obj is not None:
            action, _ = best_model_obj.predict(obs, deterministic=True)
        else:
            # 관망 모드 (전량 매도 시도)
            action = np.array([-1] * stock_dim) 

        step_res = env_test.step(action)
        if len(step_res) == 5: obs, _, term, trunc, _ = step_res; done = term or trunc
        else: obs, _, done, _ = step_res
        
        curr_state = env_test.state
        curr_cash = curr_state[0]
        curr_holdings = np.array(curr_state[1+stock_dim : 1+2*stock_dim]) 
        diff = curr_holdings - prev_holdings
        date_str = unique_dates[curr_step] if curr_step < len(unique_dates) else "End"
        
        for i, change in enumerate(diff):
            if abs(change) > 0:
                ticker = list_ticker_test[i]
                try: real_price = price_matrix.loc[date_str, ticker]
                except: real_price = 0
                if real_price > 0:
                    trade_history.append({
                        "Date": date_str, "Ticker": ticker, "Type": "매수" if change > 0 else "매도",
                        "Price": real_price, "Shares": abs(change),
                        "Cash": curr_cash, "Used_Model": best_algo_name
                    })
        prev_holdings = curr_holdings
        curr_step += 1
    
    final_val = env_test.asset_memory[-1]
    ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"      🏁 앙상블 최종 결과: {ret:.2f}%")
    
    final_summary.append({'Year': target_year, 'Model': 'ENSEMBLE', 'Return': ret})
    if trade_history:
        pd.DataFrame(trade_history).to_csv(ens_log_path, index=False, encoding='utf-8-sig')

# ==========================================
# 6. 최종 리포트
# ==========================================
print("\n" + "="*60)
print("🏆 최종 결과 요약")
print("="*60)
df_res = pd.DataFrame(final_summary)
if not df_res.empty:
    print(df_res)
    print(f"\n📂 상세 로그: {log_dir}")
else:
    print("데이터가 없습니다.")