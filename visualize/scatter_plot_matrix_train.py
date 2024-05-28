# Scatter_plot_matrix (산점도 행렬)
# 수치형 변수 간의 관계를 시각적으로 파악가능

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../data/purchase_train.csv')

# 수치형 변수만 선택
numeric_data = data[['X1', 'X2', 'X3', 'X4', 'X5', 'y']]

# Pairplot with hue set to the target variable 'y'
sns.pairplot(numeric_data, hue='y')

# 그래프를 이미지 파일로 저장
plt.savefig('pred_knn_10.png')  # 이미지 파일로 저장, 확장자를 변경하여 다른 형식으로 저장 가능

plt.show()
