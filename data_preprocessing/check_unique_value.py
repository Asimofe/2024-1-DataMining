import pandas as pd

def check_unique_values(csv_file_path, column_name):
    # CSV 파일 로드
    df = pd.read_csv(csv_file_path)
    
    # 특정 열의 고유 값 확인
    unique_values = df[column_name].unique()
    
    # 결과 출력
    print(f"'{column_name}' 열의 고유 값:")
    for value in unique_values:
        print(value)

# 사용 예시
csv_file_path = '../data/purchase_train.csv'  # CSV 파일 경로
column_name = 'X6'  # 확인하려는 열 이름

check_unique_values(csv_file_path, column_name)
