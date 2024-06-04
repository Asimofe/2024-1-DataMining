'''
Test Accuracy: 0.7113432835820895
Classification Report:
               precision    recall  f1-score   support

         0.0       0.82      0.56      0.67      1727
         1.0       0.65      0.87      0.74      1623

    accuracy                           0.71      3350
   macro avg       0.74      0.72      0.71      3350
weighted avg       0.74      0.71      0.71      3350
'''
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
file_path = 'data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# 결측값 처리 (결측값이 존재한다면 처리하는 방법, 여기서는 간단히 제거)
data.dropna(inplace=True)

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['X6'] = data['X6'].map(month_mapping)

visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
data['X11'] = data['X11'].map(visitor_mapping)

# 범주형 변수에 대해 원-핫 인코딩
data = pd.get_dummies(data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
data = data.astype(float)

# X와 y 분리
X = data.drop(columns=['y'])
y = data['y']

# 수치형 변수 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA를 사용하여 2차원으로 축소 (시각화를 위한 용도)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_components)
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df['is_outlier'] = dbscan_labels == -1

# 이상치 시각화
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[dbscan_labels != -1, 0], pca_components[dbscan_labels != -1, 1], c='b', label='Normal Data')
plt.scatter(pca_components[dbscan_labels == -1, 0], pca_components[dbscan_labels == -1, 1], c='r', label='Outliers')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('DBSCAN Outlier Detection')
plt.legend()
plt.savefig('dbscan_outliers.png')
plt.show()

# 원본 데이터에서 이상치 제거
data['is_outlier'] = pca_df['is_outlier']
cleaned_data = data[~data['is_outlier']]

# X와 y 분리 (이상치 제거 후)
X_cleaned = cleaned_data.drop(columns=['is_outlier', 'y'])
y_cleaned = cleaned_data['y']

# 오버샘플링을 통해 데이터셋의 클래스 불균형 해결
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_cleaned, y_cleaned)

# SMOTE 적용 후 데이터 시각화
pca_resampled = PCA(n_components=2)
pca_components_resampled = pca_resampled.fit_transform(X_resampled)

plt.figure(figsize=(10, 6))
plt.scatter(pca_components_resampled[y_resampled == 0, 0], pca_components_resampled[y_resampled == 0, 1], c='b', label='Class 0')
plt.scatter(pca_components_resampled[y_resampled == 1, 0], pca_components_resampled[y_resampled == 1, 1], c='r', label='Class 1')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('SMOTE Resampled Data Distribution')
plt.legend()
plt.savefig('smote_resampled_data.png')
plt.show()

# 데이터셋을 PyTorch 텐서로 변환
X_resampled = torch.tensor(X_resampled.values.astype(np.float32))  # DataFrame을 numpy로 변환 후 tensor로 변환
y_resampled = torch.tensor(y_resampled.values.astype(np.float32)).view(-1, 1)

# 데이터셋을 train과 test로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# TensorDataset과 DataLoader로 데이터 준비
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# PyTorch 딥러닝 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(X_train.shape[1])
model = nn.DataParallel(model)  # 모델을 DataParallel로 래핑하여 멀티 GPU 사용
model.to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
model.train()
for epoch in range(100):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 모델 평가
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        all_preds.extend(predictions.cpu().numpy())

accuracy = accuracy_score(y_test.cpu().numpy(), all_preds)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test.cpu().numpy(), all_preds))

# 결과를 파일로 저장
train_data = pd.concat([pd.DataFrame(X_train.cpu().numpy()), pd.DataFrame(y_train.cpu().numpy())], axis=1)
test_data = pd.concat([pd.DataFrame(X_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())], axis=1)

train_data.to_csv('train_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
