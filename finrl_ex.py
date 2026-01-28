#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import time

# =========================================================
# ★ FinRL 라이브러리 임포트 (설치 필요: pip install finrl)
# =========================================================
try:
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    # 환경 (Environment) : 주식 시장 규칙
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
except ImportError:
    print("❌ [오류] FinRL 라이브러리가 없습니다.")
    print("   터미널에 'pip install finrl'을 입력해서 설치해주세요.")
    print("   (윈도우에서 TA-Lib 설치 오류가 나면 알려주세요! 다른 방법을 알려드릴게요.)")
    exit()

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2023-12-31'
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2025-12-31'

# FinRL에서 사용하는 기술적 지표 (RSI, MACD, CCI, DX 등 자동 생성)
INDICATORS_LIST = INDICATORS 

MODELS_TO_TRAIN = ['ppo', 'a2c', 'sac', 'ddpg']

print("="*70)
print(f"🚀 FinRL 공식 예제 실험 (PPO, A2C, SAC, DDPG)")
print(f"   Note: FinRL의 'StockTradingEnv'와 'FeatureEngineer'를 사용합니다.")
print("="*70)

# ==========================================
# 2. 데이터 로드 및 전처리 (FinRL Style)
# ==========================================
current_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(current_dir, 'data', 'krx_etf_data.csv')):
    data_path = os.path.join(current_dir, 'data', 'krx_etf_data.csv')
else:
    data_path = os.path.join(current_dir, 'data', 'krx_top200_data.csv')

print(f"\n[1] 데이터 로드 및 전처리 중...")
df = pd.read_csv(data_path)
df.columns = df.columns.str.lower()

# FinRL은 'tic_name' 컬럼을 싫어해서 제거
if 'tic_name' in df.columns:
    df = df.drop(columns=['tic_name'])

# 날짜 인덱스 처리
df = df.sort_values(['date', 'tic']).reset_index(drop=True)

print(f"   FeatureEngineer로 기술적 지표 생성 중... (시간이 좀 걸립니다)")
# ★ FinRL의 강력한 기능: 자동으로 보조지표를 쫙 만들어줌
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS_LIST,
    use_vix=False, # 한국 데이터라 VIX(미국 공포지수)는 뺌
    use_turbulence=False, # 시장 이상 징후(Turbulence) 감지 기능 
    user_defined_feature=False
)

processed = fe.preprocess_data(df)

# FinRL은 데이터프레임 인덱스가 정수형이어야 함
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(pd.MultiIndex.from_product([list_date, list_ticker], names=["date", "tic"]))
# [수정] columns=["date", "tic"]을 추가해서 이름표를 붙여줌
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])
processed_full = processed_full.fillna(0)

print(f"   전처리 완료! (데이터 크기: {processed_full.shape})")

# ==========================================
# 3. 환경 설정 (FinRL Official Env)
# ==========================================
train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)

stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS_LIST)*stock_dimension

# FinRL 공식 환경 설정값
env_kwargs = {
    "hmax": 100,  # 한 번에 최대 매수/매도 수량
    "initial_amount": 10000000, # 1,000만원
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [0.0015] * stock_dimension, # 수수료
    "sell_cost_pct": [0.003] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4 # 학습 안정화를 위해 리워드 스케일링
}

e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

print(f"\n[2] 학습 환경 준비 완료 (StockTradingEnv)")

# ==========================================
# 4. 모델 학습 (FinRL DRLAgent)
# ==========================================
agent = DRLAgent(env=env_train)

# 저장 폴더
model_dir = os.path.join(current_dir, 'models_finrl')
os.makedirs(model_dir, exist_ok=True)

for algo in MODELS_TO_TRAIN:
    print(f"\n" + "-"*50)
    print(f"🔥 [{algo.upper()}] 모델 학습 시작 (FinRL Standard)")
    print("-" * 50)
    
    model_path = os.path.join(model_dir, f'finrl_{algo}')
    
    # 1. 모델 생성
    model = agent.get_model(algo)
    
    # 2. 학습
    # FinRL 예제들은 보통 5만~10만 스텝 정도 함
    trained_model = agent.train_model(
        model=model, 
        tb_log_name=algo,
        total_timesteps=50000 
    )
    
    # 3. 저장
    trained_model.save(model_path)
    print(f"✅ 저장됨: {model_path}.zip")

print("\n" + "="*70)
print("🎉 FinRL 공식 예제 실험 완료!")
print("   이제 이 모델들을 evaluate 코드로 비교해보세요.")
print("="*70)