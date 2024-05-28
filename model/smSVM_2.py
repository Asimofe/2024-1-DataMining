# SMOTE 사용 ver

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 데이터 로드
train_data = pd.read_csv('data/purchase_train.csv')
test_data = pd.read_csv('data/purchase_test.csv')

# 타겟 변수 분포 확인
print(train_data.iloc[:, -1].value_counts())

# 문자열을 숫자형으로 변환
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
train_data['X6'] = train_data['X6'].map(month_mapping)
test_data['X6'] = test_data['X6'].map(month_mapping)

# X11 열의 문자열 값을 숫자형으로 변환
visitor_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 2}
train_data['X11'] = train_data['X11'].map(visitor_mapping)
test_data['X11'] = test_data['X11'].map(visitor_mapping)

# 마지막 열이 타겟 변수라고 가정
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# 테스트 데이터에서 특성만 가져옴 (y_test가 없음)
X_test = test_data

# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 전처리 파이프라인 생성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['X11', 'X12'])
    ])

# 파이프라인 생성
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('classifier', SVC(C=0.01, kernel='linear', class_weight='balanced'))  # 클래스 가중치 균형 설정
])

# 모델 학습
pipeline.fit(X_train_resampled, y_train_resampled)

# 학습 데이터에 대한 예측
y_train_pred = pipeline.predict(X_train)

# 학습 데이터에 대한 성능 평가
print("학습 데이터 성능:")
print(classification_report(y_train, y_train_pred))

# 테스트 데이터에 대한 예측
y_pred = pipeline.predict(X_test)

# 예측 결과를 CSV 파일로 저장
predictions = pd.DataFrame({'Predicted': y_pred})
predictions.to_csv('data/smSVM_2_predictions.csv', index=False)

print("예측 결과가 'predictions.csv' 파일로 저장되었습니다.")

# 결과
# y
# 0    8412
# 1    1588
# Name: count, dtype: int64
# 학습 데이터 성능:
#               precision    recall  f1-score   support

#            0       0.90      0.68      0.78      8412
#            1       0.26      0.61      0.37      1588

#     accuracy                           0.67     10000
#    macro avg       0.58      0.64      0.57     10000
# weighted avg       0.80      0.67      0.71     10000
