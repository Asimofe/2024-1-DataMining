import pandas as pd

# 파일 경로
test_data_path = 'purchase_test.csv'
y_values_path = 'purchase_y_test.csv'

# CSV 파일 로드
test_data = pd.read_csv(test_data_path)
y_values = pd.read_csv(y_values_path)

# 'y' 값 병합
# y_values 파일에는 'y' 컬럼만 있다고 가정
# test_data의 인덱스를 기준으로 y_values를 병합
merged_data = test_data.copy()
merged_data['y'] = y_values['y']

# 병합된 데이터를 새로운 CSV 파일로 저장
merged_csv_path = 'merged_test_data.csv'
merged_data.to_csv(merged_csv_path, index=False)

merged_csv_path
