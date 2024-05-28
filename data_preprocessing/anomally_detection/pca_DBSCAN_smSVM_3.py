'''
y
0    8412
1    1588
Name: count, dtype: int64
Accuracy: 0.8137313432835821
Classification Report:
               precision    recall  f1-score   support

           0       0.84      0.78      0.81      1715
           1       0.79      0.85      0.82      1635

    accuracy                           0.81      3350
   macro avg       0.82      0.81      0.81      3350
weighted avg       0.82      0.81      0.81      3350
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 데이터 로드
file_path = '../../data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# 타겟 변수 분포 확인
print(data.iloc[:, -1].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['X6'] = data['X6'].map(month_mapping)

# X11 열의 문자열 값을 숫자형으로 변환
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
data['X11'] = data['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
data = pd.get_dummies(data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# X1부터 X12까지의 데이터 선택 및 스케일링
data_subset = data.drop(columns=['y'])  # 타겟 변수를 제외한 모든 피처 선택
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)

# PCA를 사용하여 2차원으로 축소
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_components)
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['is_outlier'] = dbscan_labels == -1

# 원본 데이터에서 이상치 제거
data['is_outlier'] = pca_df['is_outlier']
cleaned_data = data[~data['is_outlier']]

# 데이터셋 분리
X = cleaned_data.drop(columns=['is_outlier', 'y'])
y = cleaned_data['y']

# 오버샘플링을 통해 데이터셋의 클래스 불균형 해결
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 모델 학습 (클래스 가중치 조정)
clf = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)

# 성능 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 결과를 파일로 저장
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
