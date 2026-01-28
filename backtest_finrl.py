#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, DDPG

# FinRL 라이브러리
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 설정
# ==========================================
TEST_START_DATE = "2024-01-01"
INITIAL_CAPITAL = 100_000_000 
HMAX_VAL = 5000 

# 전체 모델 다 뽑아보자
MODELS_TO_EVAL = ['ppo', 'a2c', 'ddpg', 'sac']

base_dir = r"C:\Stock_AI"
data_dir = os.path.join(base_dir, 'data')
model_dir = os.path.join(base_dir, 'models_finrl')
log_dir = os.path.join(base_dir, 'trade_logs') # 로그 저장할 폴더

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print(f"💰 [System] 매매 일지(Log) 추출 모드 시작")

# ==========================================
# 2. 데이터 준비
# ==========================================
data_path = os.path.join(data_dir, 'krx_top200_data.csv')
if not os.path.exists(data_path):
    print("❌ 데이터 파일 없음!")
    exit()

print(f"📊 데이터 로드 중...")
df = pd.read_csv(data_path)
df.columns = df.columns.str.lower()
if 'tic_name' in df.columns: df = df.drop(columns=['tic_name'])

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=False, use_turbulence=False, user_defined_feature=False
)
processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(pd.MultiIndex.from_product([list_date, list_ticker], names=["date", "tic"]))
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])
processed_full = processed_full.fillna(0)

final_date = processed_full['date'].max()
test = data_split(processed_full, TEST_START_DATE, final_date)

stock_dimension = len(test.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

print(f"✅ 데이터 준비 완료. 이제 모델별로 뜯어봅니다.")

# ==========================================
# 3. 모델별 시뮬레이션 & 로그 기록
# ==========================================
env_kwargs = {
    "hmax": HMAX_VAL,
    "initial_amount": INITIAL_CAPITAL,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.0015] * stock_dimension,
    "sell_cost_pct": [0.003] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

for algo in MODELS_TO_EVAL:
    model_path = os.path.join(model_dir, f"finrl_{algo}.zip")
    if not os.path.exists(model_path):
        continue

    print(f"\n📝 [{algo.upper()}] 매매 내역 기록 중...", end=" ")
    
    try:
        if algo == 'ppo': model = PPO.load(model_path)
        elif algo == 'a2c': model = A2C.load(model_path)
        elif algo == 'ddpg': model = DDPG.load(model_path)
        elif algo == 'sac': model = SAC.load(model_path)
    except:
        print("로드 실패")
        continue

    # 환경 초기화
    env = StockTradingEnv(df=test, **env_kwargs)
    obs, _ = env.reset()
    done = False
    
    # 로그를 담을 리스트
    trade_history = []
    
    # 이전 상태 기억 (변화량 감지용)
    # env.state 구조: [현금, 주식1보유량, 주식2보유량..., 주식1가격, 주식2가격...]
    prev_holdings = np.array([0] * stock_dimension)
    
    # 날짜 트래킹용
    current_step = 0
    unique_trade_dates = test['date'].unique()
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # 행동 실행
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # --- [여기서부터 거래 내역 추출 로직] ---
        # 현재 상태 가져오기
        current_state = env.state
        current_cash = current_state[0]
        current_holdings = np.array(current_state[1 : 1+stock_dimension])
        current_prices = np.array(current_state[1+stock_dimension : 1+2*stock_dimension])
        
        # 보유량 변화 계산 (오늘보유량 - 어제보유량)
        diff_holdings = current_holdings - prev_holdings
        
        # 날짜 가져오기 (데이터 범위 안에서)
        if current_step < len(unique_trade_dates):
            today_date = unique_trade_dates[current_step]
        else:
            today_date = "End"

        # 거래가 발생한 종목만 기록
        for i, change in enumerate(diff_holdings):
            if change != 0: # 변동이 있다 = 거래했다
                ticker_name = list_ticker[i]
                price = current_prices[i]
                trade_type = "매수(BUY)" if change > 0 else "매도(SELL)"
                amount = abs(change)
                money_flow = -(change * price) # 내 돈의 변화 (매수면 마이너스, 매도면 플러스)
                
                trade_history.append({
                    "Date": today_date,
                    "Ticker": ticker_name,
                    "Type": trade_type,
                    "Price": price,
                    "Shares": amount,
                    "Total_Value": abs(money_flow),
                    "Cash_Balance": current_cash
                })
        
        # 다음 스텝을 위해 상태 업데이트
        prev_holdings = current_holdings
        current_step += 1
    
    # CSV로 저장
    df_log = pd.DataFrame(trade_history)
    csv_name = f"trade_log_{algo.upper()}.csv"
    csv_path = os.path.join(log_dir, csv_name)
    df_log.to_csv(csv_path, index=False, encoding='utf-8-sig') # 엑셀에서 한글 안 깨지게 utf-8-sig
    
    # 최종 수익률 계산
    final_asset = env.asset_memory[-1]
    ret = (final_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    print(f"완료! (수익률: {ret:.2f}%)")
    print(f"   💾 저장됨: {csv_path}")

print("\n" + "="*60)
print(f"✅ 모든 로그 저장 완료! C:\\Stock_AI\\trade_logs 폴더를 확인하세요.")
print("="*60)