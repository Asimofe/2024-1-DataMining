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
from collections import Counter

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
data = pd.get_dummies(data, columns=['X7', 'X8', 'X9', 'X10', 'X11', 'X12'], drop_first=True)

# 원핫 인코딩 후 데이터 타입 확인
print("Data types after one-hot encoding:")
print(data.dtypes)

# X1부터 X12까지의 데이터 선택 및 스케일링
data_subset = data.drop(columns=['y'])  # 타겟 변수를 제외한 모든 피처 선택
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)

# PCA를 사용하여 2차원으로 축소 (스케일링된 수치형 데이터만 사용)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data[scaled_data])

# DBSCAN으로 이상치 탐지
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(pca_components)

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
plt.savefig('dbscan_outliers.png')
plt.show()

# 원본 데이터에서 이상치 제거
data['is_outlier'] = dbscan_labels == -1
cleaned_data = data[~data['is_outlier']]

print("Class distribution after outlier removal:")
print(cleaned_data['y'].value_counts())

# 데이터셋 분리
X = cleaned_data.drop(columns=['is_outlier', 'y']).values
y = cleaned_data['y'].values

# SMOTE 적용
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Resampled dataset shape %s" % Counter(y_resampled))

# SMOTE 결과 시각화
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
X_resampled = torch.tensor(X_resampled.astype(np.float32))
y_resampled = torch.tensor(y_resampled.astype(np.float32)).view(-1, 1)

# 데이터셋 분리 (train/test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# TensorDataset과 DataLoader로 데이터 준비
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# PyTorch SVM 모델 정의
class SVM(nn.Module):
    def __init__(self, input_size):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# 힌지 손실 함수 정의
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, outputs, labels):
        labels = labels.float()
        hinge_loss = torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))
        return hinge_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SVM(X_train.shape[1])
model = nn.DataParallel(model)  # 모델을 DataParallel로 래핑하여 멀티 GPU 사용
model.to(device)

criterion = HingeLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

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
        predictions = torch.sign(outputs).cpu().numpy()
        all_preds.extend(predictions)

accuracy = accuracy_score(y_test.cpu().numpy(), all_preds)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test.cpu().numpy(), all_preds, zero_division=1))

# 결과를 파일로 저장
train_data = pd.concat([pd.DataFrame(X_train.cpu().numpy()), pd.DataFrame(y_train.cpu().numpy())], axis=1)
test_data = pd.concat([pd.DataFrame(X_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())], axis=1)

train_data.to_csv('train_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
