import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

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

# t-SNE를 사용하여 2차원으로 축소
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(scaled_data)

# t-SNE 결과를 데이터프레임으로 변환
tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])
tsne_df['is_outlier'] = data['is_outlier']

plt.figure(figsize=(10, 6))

# 정상 데이터 산점도
plt.scatter(tsne_df[~tsne_df['is_outlier']]['TSNE1'], tsne_df[~tsne_df['is_outlier']]['TSNE2'], 
            c='blue', label='Normal Data', alpha=0.5)

# 이상치 데이터 산점도
plt.scatter(tsne_df[tsne_df['is_outlier']]['TSNE1'], tsne_df[tsne_df['is_outlier']]['TSNE2'], 
            c='red', label='Outliers', alpha=0.7)

plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.title('DBSCAN Outliers Detection using t-SNE')
plt.legend()

# 시각화 결과를 파일로 저장
plt.savefig('tsne_dbscan_1_outliers.png')

# 그래프를 화면에 표시
#plt.show()
