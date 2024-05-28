'''
X1~X5 열의 데이터에 표준화, 이상치 탐지(DBSCAN), 차원축소 (2차원, PCA), 시각화
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# 데이터 로드
file_path = '../../data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# X1부터 X5까지의 데이터 선택 및 스케일링
data_subset = data[['X1', 'X2', 'X3', 'X4', 'X5']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_data)
data['is_outlier'] = dbscan_labels == -1

# PCA를 사용하여 2차원으로 축소
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# PCA 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['is_outlier'] = data['is_outlier']

# 시각화
plt.figure(figsize=(10, 6))

# 정상 데이터 산점도
plt.scatter(pca_df[~pca_df['is_outlier']]['PCA1'], pca_df[~pca_df['is_outlier']]['PCA2'], 
            c='blue', label='Normal Data', alpha=0.5)

# 이상치 데이터 산점도
plt.scatter(pca_df[pca_df['is_outlier']]['PCA1'], pca_df[pca_df['is_outlier']]['PCA2'], 
            c='red', label='Outliers', alpha=0.7)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('DBSCAN Outliers Detection using PCA')
plt.legend()

# 시각화 결과를 파일로 저장
plt.savefig('pca_dbscan_outliers.png')

# 그래프를 화면에 표시
#plt.show()
