import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 예측 결과 파일 로드
predicted_data = pd.read_csv('result/test_knn_10_predictions.csv')

# 예측 결과의 분포 시각화
plt.figure(figsize=(8, 4))
sns.countplot(x='predicted_y', data=predicted_data)
plt.title('Distribution of Predicted Categories')
plt.xlabel('Predicted Category')
plt.ylabel('Frequency')

# 그래프를 이미지 파일로 저장
plt.savefig('pred_knn_10.png')  # 이미지 파일로 저장, 확장자를 변경하여 다른 형식으로 저장 가능

# 화면에 그래프 표시
plt.show()
