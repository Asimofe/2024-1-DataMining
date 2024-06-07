'''
mlp_02.py에서 차원축소 제거
Test Accuracy: 0.8419019316493314
Classification Report:
               precision    recall  f1-score   support

         0.0       0.92      0.76      0.83      1707
         1.0       0.79      0.93      0.85      1658

    accuracy                           0.84      3365
   macro avg       0.85      0.84      0.84      3365
weighted avg       0.85      0.84      0.84      3365
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

# 데이터 로드
file_path = 'data/purchase_train.csv'  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# 타겟 변수 분포 확인
print(data['y'].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['X6'] = data['X6'].map(month_mapping)

# X11 열의 문자열 값을 숫자형으로 변환
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
data['X11'] = data['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
data = pd.get_dummies(data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
data = data.astype(float)

# 원핫 인코딩 후 데이터 타입 확인
print(data.dtypes)

# X1부터 X12까지의 데이터 선택 및 스케일링
X = data.drop(columns=['y'])
y = data['y']

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 데이터셋을 PyTorch 텐서로 변환
X_resampled = torch.tensor(X_resampled.astype(np.float32))
y_resampled = torch.tensor(y_resampled.astype(np.float32)).view(-1, 1)

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
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
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
