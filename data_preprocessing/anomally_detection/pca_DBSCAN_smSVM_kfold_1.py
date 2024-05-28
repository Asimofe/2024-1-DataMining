'''
y
0    8412
1    1588
Name: count, dtype: int64
Cross-Validation Accuracy Scores: [0.56303365 0.54394776 0.55427136 0.5678392  0.55075377]
Mean Accuracy: 0.5559691475963444
Fold Accuracy: 0.5630336514314415
Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.52      0.67      1677
           1       0.24      0.80      0.37       314

    accuracy                           0.56      1991
   macro avg       0.59      0.66      0.52      1991
weighted avg       0.82      0.56      0.62      1991

Fold Accuracy: 0.5439477649422401
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.49      0.65      1700
           1       0.22      0.83      0.35       291

    accuracy                           0.54      1991
   macro avg       0.58      0.66      0.50      1991
weighted avg       0.84      0.54      0.61      1991

Fold Accuracy: 0.5542713567839196
Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.50      0.65      1670
           1       0.24      0.82      0.37       320

    accuracy                           0.55      1990
   macro avg       0.59      0.66      0.51      1990
weighted avg       0.83      0.55      0.61      1990

Fold Accuracy: 0.5678391959798995
Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.53      0.67      1649
           1       0.25      0.76      0.38       341

    accuracy                           0.57      1990
   macro avg       0.58      0.64      0.52      1990
weighted avg       0.80      0.57      0.62      1990

Fold Accuracy: 0.5507537688442211
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.51      0.66      1679
           1       0.23      0.77      0.35       311

    accuracy                           0.55      1990
   macro avg       0.58      0.64      0.50      1990
weighted avg       0.82      0.55      0.61      1990
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, make_scorer

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

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)

# 교차 검증 정확도 계산
accuracy_scorer = make_scorer(accuracy_score)
cross_val_scores = cross_val_score(model, X, y, cv=kf, scoring=accuracy_scorer)

# 교차 검증 결과 출력
print("Cross-Validation Accuracy Scores:", cross_val_scores)
print("Mean Accuracy:", np.mean(cross_val_scores))

# 마지막 Fold로 모델 학습 및 평가
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 성능 평가
    print("Fold Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 결과를 파일로 저장 (마지막 Fold 기준)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('train_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
