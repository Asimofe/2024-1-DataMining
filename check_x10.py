import pandas as pd

# 데이터 로드
train_data = pd.read_csv('data/purchase_train.csv')
test_data = pd.read_csv('data/purchase_test.csv')

# 훈련 데이터와 테스트 데이터의 X10 열에서 고유값 출력
train_unique_x10 = train_data['X10'].unique()
test_unique_x10 = test_data['X10'].unique()

print("Unique X10 in Train Data:", train_unique_x10)
print("Unique X10 in Test Data:", test_unique_x10)

# 테스트 데이터에만 있는 고유값 확인
new_categories_x10 = set(test_unique_x10) - set(train_unique_x10)
print("New Categories in X10:", new_categories_x10)

"""
 train_data, 열 X10의 고유값   1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16    18 19 20
 test_data, 열 X10의 고유값    1 2 3 4 5 6 7 8 9 10 11    13 14    16 17 18 19 20
 훈련 데이터에 17이 없어 테스트 데이터에서 새로운 값으로 인식하여 에러 발생 
 원-핫 인코딩에서 문제 발생
"""