import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 데이터 로드
train_file_path = '../../data/purchase_train.csv'  # 훈련 데이터 파일 경로
test_file_path = '../../data/purchase_test.csv'    # 테스트 데이터 파일 경로
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 타겟 변수 분포 확인
print(train_data['y'].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}

for df in [train_data, test_data]:
    df['X6'] = df['X6'].map(month_mapping)
    df['X11'] = df['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
train_data = pd.get_dummies(train_data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# 훈련 데이터와 테스트 데이터의 열을 동일하게 맞추기
missing_cols_train = set(test_data.columns) - set(train_data.columns)
missing_cols_test = set(train_data.columns) - set(test_data.columns)

for col in missing_cols_train:
    train_data[col] = 0
for col in missing_cols_test:
    test_data[col] = 0

# 열 순서 맞추기
train_data = train_data[test_data.columns]

# 훈련 데이터 전처리
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# PCA를 사용하여 2차원으로 축소
pca = PCA(n_components=2)
pca_train_components = pca.fit_transform(X_train_scaled)

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_train_components)
pca_train_df = pd.DataFrame(data=pca_train_components, columns=['PCA1', 'PCA2'])
pca_train_df['is_outlier'] = dbscan_labels == -1

# 원본 데이터에서 이상치 제거
train_data['is_outlier'] = pca_train_df['is_outlier']
cleaned_train_data = train_data[~train_data['is_outlier']]

# 데이터셋 분리
X_train_cleaned = cleaned_train_data.drop(columns=['is_outlier', 'y'])
y_train_cleaned = cleaned_train_data['y']

# 스케일링
X_train_cleaned_scaled = scaler.fit_transform(X_train_cleaned)

# 모델 학습 (클래스 가중치 조정)
clf = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
clf.fit(X_train_cleaned_scaled, y_train_cleaned)

# 테스트 데이터 전처리
X_test = test_data.drop(columns=['y'])
y_test = test_data['y']
X_test_scaled = scaler.transform(X_test)

# 예측
y_test_pred = clf.predict(X_test_scaled)

# 성능 평가
print("Test Data Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Data Classification Report:\n", classification_report(y_test, y_test_pred))

# 결과를 파일로 저장
train_data.to_csv('train_1.csv', index=False)
test_data['predicted_y'] = y_test_pred
test_data.to_csv('test_1.csv', index=False)
