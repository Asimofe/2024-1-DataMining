'''
Internal Test Accuracy: 0.7585
Internal Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.85      0.85      1682
         1.0       0.26      0.29      0.28       318

    accuracy                           0.76      2000
   macro avg       0.56      0.57      0.57      2000
weighted avg       0.77      0.76      0.76      2000

Final Test Accuracy: 0.7527223230490018
Final Classification Report:
               precision    recall  f1-score   support

         0.0       0.88      0.82      0.85      1884
         1.0       0.26      0.37      0.30       320

    accuracy                           0.75      2204
   macro avg       0.57      0.59      0.58      2204
weighted avg       0.79      0.75      0.77      2204
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

# 훈련 데이터 로드
train_file_path = 'data/purchase_train.csv'  # 훈련 데이터 파일 경로를 변경하세요
train_data = pd.read_csv(train_file_path)

# 타겟 변수 분포 확인
print(train_data['y'].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
train_data['X6'] = train_data['X6'].map(month_mapping)

# X11 열의 문자열 값을 숫자형으로 변환
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
train_data['X11'] = train_data['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
train_data = pd.get_dummies(train_data, columns=['X7', 'X8', 'X9', 'X10', 'X11', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
train_data = train_data.astype(float)

# 원핫 인코딩 후 데이터 타입 확인
print(train_data.dtypes)

# X와 y 선택 및 스케일링
X = train_data.drop(columns=['y'])
y = train_data['y']

# 데이터를 훈련 데이터와 내부 테스트 데이터로 분할
X_train, X_internal_test, y_train, y_internal_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_internal_test_scaled = scaler.transform(X_internal_test)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# 데이터셋을 PyTorch 텐서로 변환
X_resampled = torch.tensor(X_resampled.astype(np.float32))
y_resampled = torch.tensor(y_resampled.astype(np.float32)).view(-1, 1)

# TensorDataset과 DataLoader로 데이터 준비
train_dataset = TensorDataset(X_resampled, y_resampled)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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

model = MLP(X_resampled.shape[1])
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

# 내부 테스트 데이터셋 평가
X_internal_test_tensor = torch.tensor(X_internal_test_scaled.astype(np.float32))
y_internal_test_tensor = torch.tensor(y_internal_test.values.astype(np.float32)).view(-1, 1)

internal_test_dataset = TensorDataset(X_internal_test_tensor, y_internal_test_tensor)
internal_test_loader = DataLoader(internal_test_dataset, batch_size=32, shuffle=False)

model.eval()
internal_preds = []
with torch.no_grad():
    for inputs, labels in internal_test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        internal_preds.extend(predictions.cpu().numpy())

internal_accuracy = accuracy_score(y_internal_test_tensor.cpu().numpy(), internal_preds)
print(f"Internal Test Accuracy: {internal_accuracy}")
print("Internal Classification Report:\n", classification_report(y_internal_test_tensor.cpu().numpy(), internal_preds))

# 모델 평가를 위한 별도 테스트 데이터 로드 및 전처리
test_data_path = 'data/purchase_test.csv'  # 테스트 데이터 파일 경로를 변경하세요
test_data_with_y = pd.read_csv(test_data_path)

# y 열이 있는지 확인
if 'y' in test_data_with_y.columns:
    print("'y' column is found in the test data.")
else:
    raise KeyError("The 'y' column is not found in the test data.")

# 문자열을 숫자형으로 변환
test_data_with_y['X6'] = test_data_with_y['X6'].map(month_mapping)
test_data_with_y['X11'] = test_data_with_y['X11'].map(visitor_mapping)

# 원 핫 인코딩
test_data_with_y = pd.get_dummies(test_data_with_y, columns=['X7', 'X8', 'X9', 'X10', 'X11', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
test_data_with_y = test_data_with_y.astype(float)

print(test_data_with_y.dtypes)

# X와 y 분리
X_test_final = test_data_with_y.drop(columns=['y'])
y_test_final = test_data_with_y['y']

# 훈련 데이터와 테스트 데이터의 컬럼을 일치시킴
missing_cols = set(X_train.columns) - set(X_test_final.columns)
for col in missing_cols:
    X_test_final[col] = 0
X_test_final = X_test_final[X_train.columns]

# 스케일링
X_test_final_scaled = scaler.transform(X_test_final)

# 데이터셋을 PyTorch 텐서로 변환
X_test_final_tensor = torch.tensor(X_test_final_scaled.astype(np.float32))
y_test_final_tensor = torch.tensor(y_test_final.values.astype(np.float32)).view(-1, 1)

# 평가 데이터로 모델 성능 평가
test_final_dataset = TensorDataset(X_test_final_tensor, y_test_final_tensor)
test_final_loader = DataLoader(test_final_dataset, batch_size=32, shuffle=False)

model.eval()
all_final_preds = []
with torch.no_grad():
    for inputs, labels in test_final_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        all_final_preds.extend(predictions.cpu().numpy())

final_accuracy = accuracy_score(y_test_final_tensor.cpu().numpy(), all_final_preds)
print(f"Final Test Accuracy: {final_accuracy}")
print("Final Classification Report:\n", classification_report(y_test_final_tensor.cpu().numpy(), all_final_preds))
