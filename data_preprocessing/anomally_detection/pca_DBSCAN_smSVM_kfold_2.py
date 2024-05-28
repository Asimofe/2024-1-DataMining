import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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

# 원핫 인코딩 후 데이터 타입 확인
print(data.dtypes)

# X1부터 X12까지의 데이터 선택 및 스케일링
data_subset = data.drop(columns=['y'])  # 타겟 변수를 제외한 모든 피처 선택

# 모든 열을 float 타입으로 변환
data_subset = data_subset.astype(float)

# 스케일링
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
X = cleaned_data.drop(columns=['is_outlier', 'y']).values
y = cleaned_data['y'].values

# 데이터셋을 PyTorch 텐서로 변환
X = torch.tensor(X.astype(np.float32))  # 데이터 타입을 float32로 변환
y = torch.tensor(y.astype(np.float32)).view(-1, 1)

# TensorDataset과 DataLoader로 데이터 준비
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 교차 검증을 통한 성능 평가
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    accuracies.append(accuracy)

    print(f"Fold Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test.cpu().numpy(), all_preds))

# 교차 검증 결과 출력
print("Cross-Validation Accuracy Scores:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))

# 결과를 파일로 저장 (마지막 Fold 기준)
train_data = pd.concat([pd.DataFrame(X_train.cpu().numpy()), pd.DataFrame(y_train.cpu().numpy())], axis=1)
test_data = pd.concat([pd.DataFrame(X_test.cpu().numpy()), pd.DataFrame(y_test.cpu().numpy())], axis=1)

train_data.to_csv('train_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
