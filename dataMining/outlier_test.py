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
file_path = 'data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# 타겟 변수 분포 확인
print("Class distribution in the original data:")
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

# 원핫 인코딩 후 데이터 타입 확인
print("Data types after one-hot encoding:")
#print(data.dtypes)

# 수치형 데이터 열 이름 추출
numeric_features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']

# 원핫인코딩된 열과 수치형 열 구분
onehot_features = data.columns.difference(numeric_features + ['y'])
print(onehot_features)

# 수치형 데이터만 스케일링
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# PCA를 사용하여 2차원으로 축소 (스케일링된 수치형 데이터만 사용)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data[numeric_features])

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_components)
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['is_outlier'] = dbscan_labels == -1

# 이상치 비율 확인
outlier_ratio = np.sum(dbscan_labels == -1) / len(dbscan_labels)
print(f"Outlier ratio: {outlier_ratio:.2f}")

# DBSCAN 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[dbscan_labels != -1, 0], pca_components[dbscan_labels != -1, 1], c='b', label='Normal Data')
plt.scatter(pca_components[dbscan_labels == -1, 0], pca_components[dbscan_labels == -1, 1], c='r', label='Outliers')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('DBSCAN Outlier Detection')
plt.legend()
plt.savefig('dbscan_outliers_test.png')
plt.show()