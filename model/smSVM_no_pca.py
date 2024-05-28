import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# 전처리된 데이터 로드
train_data_processed = pd.read_csv('data/processed_train_data.csv')
test_data_processed = pd.read_csv('data/processed_test_data.csv')

# 타겟 변수 로드
train_data = pd.read_csv('data/purchase_train.csv')
y_train = train_data.iloc[:, -1]

# PCA를 사용하지 않는 파이프라인
pipeline_no_pca = Pipeline(steps=[
    ('scaler', StandardScaler()),  # 다시 스케일링 적용
    ('classifier', SVC(kernel='linear', class_weight='balanced'))  # 클래스 가중치 균형 설정
])

# 다양한 C 값을 시도
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}

# GridSearchCV를 사용하여 최적의 C 값 찾기
grid_search_no_pca = GridSearchCV(pipeline_no_pca, param_grid, cv=5, scoring='f1_macro')

# 모델 학습
grid_search_no_pca.fit(train_data_processed, y_train)

# 최적의 파라미터 출력
print("최적의 파라미터 (PCA 미사용):", grid_search_no_pca.best_params_)

# 최적의 모델로 학습 데이터에 대한 예측
y_train_pred_no_pca = grid_search_no_pca.predict(train_data_processed)

# 학습 데이터에 대한 성능 평가
print("학습 데이터 성능 (PCA 미사용):")
print(classification_report(y_train, y_train_pred_no_pca))

# 최적의 모델로 테스트 데이터에 대한 예측
y_pred_no_pca = grid_search_no_pca.predict(test_data_processed)

# 예측 결과를 CSV 파일로 저장 (PCA 미사용)
predictions_no_pca = pd.DataFrame({'Predicted': y_pred_no_pca})
predictions_no_pca.to_csv('data/predictions_no_pca.csv', index=False)

print("예측 결과가 'predictions_no_pca.csv' 파일로 저장되었습니다.")
