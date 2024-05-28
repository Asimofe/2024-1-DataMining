import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# 데이터 로드
train_data = pd.read_csv('data/purchase_train.csv')
test_data = pd.read_csv('data/purchase_test.csv')

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

# 수치형 변수와 범주형 변수 구분
numeric_features = ['X1', 'X2', 'X3', 'X4', 'X5']
ordinal_features = ['X6']  # 순서가 있는 데이터 (페이지 방문 월)
categorical_features = ['X7', 'X8', 'X9', 'X10', 'X11', 'X12']  # 순서가 없는 데이터

# 전처리 파이프라인 생성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('ord', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ]), ordinal_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# PCA를 사용하지 않는 파이프라인
pipeline_no_pca = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', class_weight='balanced'))  # 클래스 가중치 균형 설정
])

# PCA를 사용하는 파이프라인
n_components = 10  # 원하는 주성분 수 설정
pipeline_with_pca = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components)),
    ('classifier', SVC(kernel='linear', class_weight='balanced'))  # 클래스 가중치 균형 설정
])

# 다양한 C 값을 시도
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}

# GridSearchCV를 사용하여 최적의 C 값 찾기
grid_search_no_pca = GridSearchCV(pipeline_no_pca, param_grid, cv=5, scoring='f1_macro')
grid_search_with_pca = GridSearchCV(pipeline_with_pca, param_grid, cv=5, scoring='f1_macro')

# 모델 학습
grid_search_no_pca.fit(X_train_resampled, y_train_resampled)
grid_search_with_pca.fit(X_train_resampled, y_train_resampled)

# 최적의 파라미터 출력
print("최적의 파라미터 (PCA 미사용):", grid_search_no_pca.best_params_)
print("최적의 파라미터 (PCA 사용):", grid_search_with_pca.best_params_)

# 최적의 모델로 학습 데이터에 대한 예측
y_train_pred_no_pca = grid_search_no_pca.predict(X_train)
y_train_pred_with_pca = grid_search_with_pca.predict(X_train)

# 학습 데이터에 대한 성능 평가
print("학습 데이터 성능 (PCA 미사용):")
print(classification_report(y_train, y_train_pred_no_pca))

print("학습 데이터 성능 (PCA 사용):")
print(classification_report(y_train, y_train_pred_with_pca))

# 최적의 모델로 테스트 데이터에 대한 예측
y_pred_no_pca = grid_search_no_pca.predict(X_test)
y_pred_with_pca = grid_search_with_pca.predict(X_test)

# 예측 결과를 CSV 파일로 저장 (PCA 미사용)
predictions_no_pca = pd.DataFrame({'Predicted': y_pred_no_pca})
predictions_no_pca.to_csv('/mnt/data/predictions_no_pca.csv', index=False)

# 예측 결과를 CSV 파일로 저장 (PCA 사용)
predictions_with_pca = pd.DataFrame({'Predicted': y_pred_with_pca})
predictions_with_pca.to_csv('/mnt/data/predictions_with_pca.csv', index=False)

print("예측 결과가 'predictions_no_pca.csv'와 'predictions_with_pca.csv' 파일로 저장되었습니다.")
