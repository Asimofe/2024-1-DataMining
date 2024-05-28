import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 데이터 로드
file_path = '../../data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# X1부터 X5까지의 데이터 선택 및 스케일링
data_subset = data[['X1', 'X2', 'X3', 'X4', 'X5']]
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
cleaned_data = data[~pca_df['is_outlier']]

# 제거된 데이터에 대해 다시 스케일링
scaled_cleaned_data = scaler.fit_transform(cleaned_data[['X1', 'X2', 'X3', 'X4', 'X5']])

# PCA를 사용하여 제거된 데이터 다시 축소
cleaned_pca_components = pca.fit_transform(scaled_cleaned_data)
cleaned_pca_df = pd.DataFrame(data=cleaned_pca_components, columns=['PCA1', 'PCA2'])

# 결과 시각화
plt.figure(figsize=(10, 6))

# 정상 데이터 산점도
plt.scatter(cleaned_pca_df['PCA1'], cleaned_pca_df['PCA2'], c='blue', label='Normal Data', alpha=0.5)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('DBSCAN Outliers Removed and PCA Visualization')
plt.legend()

# 시각화 결과를 파일로 저장
plt.savefig('pca_dbscan_outliers_removed.png')

# 그래프를 화면에 표시
# plt.show()
