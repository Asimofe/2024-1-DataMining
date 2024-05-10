import pandas as pd

# 데이터 파일 로드
train_data = pd.read_csv('data/purchase_train.csv')
# pred_knn_5 = pd.read_csv('result/test_knn_5_predictions.csv')
# pred_knn_10 = pd.read_csv('result/test_knn_10_predictions.csv')  
#pred_lr = pd.read_csv('result/test_logit_reg_predictions.csv')  

# 'y' 열의 값이 1인 행만 필터링
filtered_data = train_data[train_data['y'] == 1]
# filtered_data1 = pred_knn_5[pred_knn_5['predicted_y'] == 1]
# filtered_data2 = pred_knn_10[pred_knn_10['predicted_y'] == 1]

# 'y' 열의 값이 1인 행만 필터링
#filtered_data2 = pred_lr[pred_lr['predicted_y'] == 1]

# 결과 출력
print(filtered_data)
# print('pred_knn_5')
# print(filtered_data1)

# print('pred_knn_10')
# print(filtered_data2)