# KNN으로 테스트 데이터의 y 예측
# K 값을 변화하면서 테스트 해보기

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기 (예제에서는 'data.csv'로 가정)
train_data = pd.read_csv('data/purchase_train.csv')
test_data = pd.read_csv('data/purchase_test.csv')

# 훈련 데이터셋의 feature와 target 분리
X_train = train_data.drop('y', axis=1)  # 모든 입력변수 선택
y_train = train_data['y']  # 출력변수 (구매여부)

# 테스트 데이터셋의 feature와 target 분리 (테스트 데이터에 'y' 열이 있다고 가정)X
X_test = test_data

# 범주형 변수와 수치형 변수를 처리하는 파이프라인 생성
categorical_features = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12']
numeric_features = ['X1', 'X2', 'X3', 'X4', 'X5']

# 데이터 전처리
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 파이프라인 구성
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier(n_neighbors=10))])

# 모델 학습
pipeline.fit(X_train, y_train)

# 예측
y_pred = pipeline.predict(X_test)

# 성능 평가
#accuracy = accuracy_score(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred)
#report = classification_report(y_test, y_pred)

#print("Accuracy:", accuracy)
#print("Confusion Matrix:\n", conf_matrix)
#print("Classification Report:\n", report)

# 예측 결과를 파일로 저장
test_data['predicted_y'] = y_pred
test_data.to_csv('test_knn_10_predictions.csv', index=False)