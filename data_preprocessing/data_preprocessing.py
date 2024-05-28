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
    ], sparse_threshold=0)  # sparse_threshold=0: 밀집 행렬로 변환

# 전처리 파이프라인을 통해 데이터 변환
X_train_processed = preprocessor.fit_transform(X_train_resampled)
X_test_processed = preprocessor.transform(X_test)

# 전처리된 데이터를 데이터프레임으로 변환
train_processed_df = pd.DataFrame(X_train_processed)
test_processed_df = pd.DataFrame(X_test_processed)

# 전처리된 데이터를 CSV 파일로 저장
train_processed_df.to_csv('data/processed_train_data.csv', index=False)
test_processed_df.to_csv('data/processed_test_data.csv', index=False)

print("전처리된 데이터가 'processed_train_data.csv'와 'processed_test_data.csv' 파일로 저장되었습니다.")
