import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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
train_data = pd.get_dummies(train_data, columns=['X7', 'X8', 'X9', 'X10','X11', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
train_data = train_data.astype(float)

# 원핫 인코딩 후 데이터 타입 확인
# print(train_data.dtypes)

# X와 y 선택
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 랜덤 포레스트 모델 정의
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 모델 학습
model.fit(X_resampled, y_resampled)

# 내부 테스트 데이터셋 평가를 위한 별도 분할
X_internal_train, X_internal_test, y_internal_train, y_internal_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# 내부 테스트 데이터 예측
y_internal_pred = model.predict(X_internal_test)

# 내부 테스트 데이터 성능 평가
internal_accuracy = accuracy_score(y_internal_test, y_internal_pred)
print(f"Internal Test Accuracy: {internal_accuracy}")
print("Internal Classification Report:\n", classification_report(y_internal_test, y_internal_pred))

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
test_data_with_y = pd.get_dummies(test_data_with_y, columns=['X7', 'X8', 'X9', 'X10','X11', 'X12'], drop_first=True)

# 모든 열을 float 타입으로 변환
test_data_with_y = test_data_with_y.astype(float)

# print(test_data_with_y.dtypes)

# X와 y 분리
X_test_final = test_data_with_y.drop(columns=['y'])
y_test_final = test_data_with_y['y']

# 훈련 데이터와 테스트 데이터의 컬럼을 일치시킴
missing_cols = set(X_train.columns) - set(X_test_final.columns)
for col in missing_cols:
    X_test_final[col] = 0
X_test_final = X_test_final[X_train.columns]

# 최종 테스트 데이터 예측
y_final_pred = model.predict(X_test_final)

# 최종 테스트 데이터 성능 평가
final_accuracy = accuracy_score(y_test_final, y_final_pred)
print(f"Final Test Accuracy: {final_accuracy}")
print("Final Classification Report:\n", classification_report(y_test_final, y_final_pred))
