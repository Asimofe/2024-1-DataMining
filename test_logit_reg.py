# 로지스틱 회귀로 테스트 데이터의 y 예측

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 데이터 로드
train_data = pd.read_csv('data/purchase_train.csv')
test_data = pd.read_csv('data/purchase_test.csv')

print('train_data')
print(train_data.head())

print('test_data')
print(test_data.head())

# 입력 변수와 출력 변수 분리
X_train = train_data.drop('y', axis=1)
y_train = train_data['y']
X_test = test_data  # 테스트 데이터에는 'y' 열이 없다고 가정

# 범주형 변수와 수치형 변수를 처리하는 파이프라인 생성
categorical_features = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']
numeric_features = ['X1', 'X2', 'X3', 'X4', 'X5']

# 데이터 전처리
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 로지스틱 회귀 모델을 사용한 파이프라인 생성
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='liblinear'))])

# 모델 학습
clf.fit(X_train, y_train)

# 예측 수행
y_pred = clf.predict(X_test)

# 성능 평가 (테스트 데이터셋에 대한 정답이 필요)
# print(classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# 예측 결과를 파일로 저장
test_data['predicted_y'] = y_pred
test_data.to_csv('test_logit_reg_predictions.csv', index=False)
