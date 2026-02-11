import pandas as pd

# Parquet 파일 읽기
file_path = r'C:\Stock_AI\data_parquet\0120G0.parquet'
df = pd.read_parquet(file_path)

# 데이터 확인
print(df.head())      # 상단 5행 출력
print(df.columns)     # 어떤 지표들이 들어있는지 확인
print(df.info())      # 데이터 타입과 누락된 값 확인