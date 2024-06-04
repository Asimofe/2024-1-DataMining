import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드
train_file_path = 'data/purchase_train.csv'  # 훈련 데이터 파일 경로를 변경하세요
test_file_path = 'data/purchase_test.csv'  # 테스트 데이터 파일 경로를 변경하세요

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 타겟 변수 분포 확인
print(train_data['y'].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
train_data['X6'] = train_data['X6'].map(month_mapping)
test_data['X6'] = test_data['X6'].map(month_mapping)

# X11 열의 문자열 값을 숫자형으로 변환
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
train_data['X11'] = train_data['X11'].map(visitor_mapping)
test_data['X11'] = test_data['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
train_data = pd.get_dummies(train_data, columns=['X7', 'X8', 'X9', 'X10','X12'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['X7', 'X8', 'X9', 'X10','X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
train_data = train_data.astype(float)
test_data = test_data.astype(float)

# 훈련 데이터와 테스트 데이터의 열을 동일하게 맞추기
missing_cols_train = set(test_data.columns) - set(train_data.columns)
for col in missing_cols_train:
    train_data[col] = 0

missing_cols_test = set(train_data.columns) - set(test_data.columns)
for col in missing_cols_test:
    test_data[col] = 0

train_data = train_data[sorted(train_data.columns)]
test_data = test_data[sorted(test_data.columns)]

# X와 y 선택 및 스케일링
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
X_test = test_data.drop(columns=['y'])
y_test = test_data['y']

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# 데이터셋을 PyTorch 텐서로 변환
X_resampled = torch.tensor(X_resampled.astype(np.float32))
y_resampled = torch.tensor(y_resampled.astype(np.float32)).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
y_test_tensor = torch.tensor(y_test.astype(np.float32)).view(-1, 1)

# 데이터셋을 train과 validation으로 나누기
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# TensorDataset과 DataLoader로 데이터 준비
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
        labels = 2 * labels - 1  # 라벨을 -1 또는 1로 변환
        hinge_loss = torch.mean(torch.clamp(1 - outputs * labels, min=0))
        return hinge_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SVM(X_train.shape[1])
model = nn.DataParallel(model)  # 모델을 DataParallel로 래핑하여 멀티 GPU 사용
model.to(device)

# 모델 가중치 초기화
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(weights_init)

criterion = HingeLoss().to(device)
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

# 모델 평가 (검증 데이터셋)
model.eval()
val_preds = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = torch.sign(outputs)
        val_preds.extend(predictions.cpu().numpy())

val_accuracy = accuracy_score(y_val.cpu().numpy(), val_preds)
print(f"Validation Accuracy: {val_accuracy}")
print("Validation Classification Report:\n", classification_report(y_val.cpu().numpy(), val_preds))

# 모델 평가 (테스트 데이터셋)
test_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = torch.sign(outputs)
        test_preds.extend(predictions.cpu().numpy())

test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), test_preds)
print(f"Test Accuracy: {test_accuracy}")
print("Test Classification Report:\n", classification_report(y_test_tensor.cpu().numpy(), test_preds))

# 결과를 파일로 저장
train_data = pd.concat([pd.DataFrame(X_train.cpu().numpy()), pd.DataFrame(y_train.cpu().numpy())], axis=1)
val_data = pd.concat([pd.DataFrame(X_val.cpu().numpy()), pd.DataFrame(y_val.cpu().numpy())], axis=1)
test_data = pd.concat([pd.DataFrame(X_test_tensor.cpu().numpy()), pd.DataFrame(y_test_tensor.cpu().numpy())], axis=1)

train_data.to_csv('train_1.csv', index=False)
val_data.to_csv('val_1.csv', index=False)
test_data.to_csv('test_1.csv', index=False)
