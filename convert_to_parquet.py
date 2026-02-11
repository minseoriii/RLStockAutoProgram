import os
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS

# 1. 경로 설정
base_dir = r"C:\Stock_AI"
csv_path = os.path.join(base_dir, 'data', 'minute_data_all.csv')
save_dir = os.path.join(base_dir, 'data_parquet')

if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

# 2. 전처리 엔진
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS)
def split_and_convert():
    print("🧹 데이터 재세탁 시작! 이번엔 컬럼 순서 확실히 잡자!")
    full_df = pd.read_csv(csv_path, low_memory=False)
    
    # [★수정] 민서 데이터 실제 순서: 날짜, 종목코드, 시, 고, 저, 종, 거
    full_df.columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    
    # 🛠️ 숫자 변환 (tic은 종목코드니까 빼고 나머지 고치기)
    cols_to_fix = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_fix:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    
    full_df = full_df.dropna(subset=cols_to_fix)
    full_df[cols_to_fix] = full_df[cols_to_fix].abs()
    
    # 이제 tic(종목코드)별로 쪼개면 파일 이름이 '185750.parquet' 처럼 예쁘게 나옴!
    unique_tics = full_df['tic'].unique()

    print(f"✅ 총 {len(unique_tics)}개 종목 확인됨. 세척 시작!")

    for tic in unique_tics:
        try:
            print(f"📦 {tic} 처리 중...", end=" ")
            df_tic = full_df[full_df['tic'] == tic].copy()
            
            # [추가] 중복된 시간 제거 (에러의 주범!)
            df_tic = df_tic.drop_duplicates(subset=['date'])
            
            # [추가] 데이터가 너무 적으면 지표 계산이 안 됨 (최소 50개 이상)
            if len(df_tic) < 50:
                print(f"⏩ 데이터 너무 적음 ({len(df_tic)}개), 건너뜀")
                continue
            
            # 시간 순 정렬
            df_tic = df_tic.sort_values('date').reset_index(drop=True)
            
            # 기술적 지표 계산
            # preprocess_data 내부에서 발생하는 인덱스 에러 방지를 위해 copy() 사용
            df_tic = fe.preprocess_data(df_tic)
            
            # 지표 계산 후 필수 컬럼이 있는지 확인 (에러 방지)
            if 'boll_ub' not in df_tic.columns:
                print(f"❌ 지표 계산 실패, 건너뜀")
                continue

            # 날짜 파싱 및 시간/분 추출
            df_tic['date_dt'] = pd.to_datetime(df_tic['date'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
            df_tic = df_tic.dropna(subset=['date_dt'])
            
            df_tic['hour'] = df_tic['date_dt'].dt.hour
            df_tic['minute'] = df_tic['date_dt'].dt.minute
            df_tic = df_tic.drop(columns=['date_dt'])
            
            # 저장
            save_path = os.path.join(save_dir, f"{tic}.parquet")
            df_tic.to_parquet(save_path, compression='snappy', index=False)
            print(f"✅ 완료 ({len(df_tic)}행)")
            
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            continue
            
        del df_tic
        
    print("\n🏁 모든 종목 세척 완료! 이제 진짜 학습 준비 끝!")

if __name__ == "__main__":
    split_and_convert()