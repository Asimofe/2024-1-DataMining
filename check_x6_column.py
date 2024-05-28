import pandas as pd

# 데이터 로드
train_data = pd.read_csv('data/purchase_train.csv')

# X6 열의 고유 값과 각 값의 개수 확인
X6_values_count = train_data['X6'].value_counts(dropna=False)

print("X6 열의 값 분포:")
print(X6_values_count)


# 데이터 로드
test_data = pd.read_csv('data/purchase_test.csv')

# X6 열의 고유 값과 각 값의 개수 확인
X6_values_count = test_data['X6'].value_counts(dropna=False)

print("X6 열의 값 분포:")
print(X6_values_count)
