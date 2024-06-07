import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 데이터 로드
train_file_path = '../data/purchase_train.csv'  # 훈련 데이터 파일 경로
test_file_path = '../data/purchase_test.csv'    # 테스트 데이터 파일 경로
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 타겟 변수 분포 확인
print(train_data['y'].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}

for df in [train_data, test_data]:
    df['X6'] = df['X6'].map(month_mapping)
    df['X11'] = df['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
train_data = pd.get_dummies(train_data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# 훈련 데이터와 테스트 데이터의 열을 동일하게 맞추기
missing_cols_train = set(test_data.columns) - set(train_data.columns)
missing_cols_test = set(train_data.columns) - set(test_data.columns)

for col in missing_cols_train:
    train_data[col] = 0
for col in missing_cols_test:
    test_data[col] = 0

# 열 순서 맞추기
train_data = train_data[test_data.columns]

# 훈련 데이터 전처리
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터 전처리
X_test = test_data.drop(columns=['y'])
y_test = test_data['y']
X_test_scaled = scaler.transform(X_test)

# 훈련 데이터셋과 테스트 데이터셋 분리
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_split, y_train_split)

# 모델 학습 (클래스 가중치 조정)
clf = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# 내부 검증 데이터에 대한 예측
y_val_pred = clf.predict(X_val_split)

# 성능 평가
print("Internal Validation Accuracy:", accuracy_score(y_val_split, y_val_pred))
print("Internal Validation Classification Report:\n", classification_report(y_val_split, y_val_pred))

# 테스트 데이터에 대한 예측
y_test_pred = clf.predict(X_test_scaled)

# 테스트 데이터 성능 평가
print("Test Data Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Data Classification Report:\n", classification_report(y_test, y_test_pred))
