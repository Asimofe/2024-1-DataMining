import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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

# 합쳐진 데이터프레임 생성
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'
combined_data = pd.concat([train_data, test_data])

# 문자열을 숫자형으로 변환
combined_data['X6'] = combined_data['X6'].map(month_mapping)
combined_data['X11'] = combined_data['X11'].map(visitor_mapping)

# X7부터 X12까지의 데이터 원 핫 인코딩
combined_data = pd.get_dummies(combined_data, columns=['X7', 'X8', 'X9', 'X10', 'X12'], drop_first=True)

# 'dataset' 열을 제거하고 나머지 열을 float 타입으로 변환
combined_data = combined_data.drop(columns=['dataset'])
combined_data = combined_data.astype(float)

# 데이터셋 분리
train_data = combined_data[combined_data.index.isin(train_data.index)]
test_data = combined_data[combined_data.index.isin(test_data.index)]

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

# 모델 학습 (클래스 가중치 조정)
clf = SVC(C=1.0, kernel='linear', class_weight='balanced', random_state=42)
clf.fit(X_train_split, y_train_split)

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
