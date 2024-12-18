import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로 설정
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'

# 데이터 로드
data = pd.read_excel(file_path)

# 데이터 전처리
data = data[['label', 'tear_uric_acid']].dropna()  #


data = data[data['label'] != 0]
data['label'] = data['label'].replace({1: 0, 2: 1})  #

# 특징과 레이블 분리
features = data[['tear_uric_acid']]
labels = data['label']

# StandardScaler로 피처 표준화
scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=['tear_uric_acid'])


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# XGBoost DMatrix
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 5,  #
    'colsample_bytree': 0.8  #
}

# 모델 학습
xgboost_model = xgb.train(params, train_data, num_boost_round=100)

# 예측
y_pred = xgboost_model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)  # 확률을 이진 분류로 변환

# Confusion Matrix 계산 및 출력
label_names = ['Gout w/o treatment', 'Gout w/ treatment']
cm = confusion_matrix(y_test, y_pred_binary)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("Confusion Matrix (Percentage by Row):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary, target_names=label_names))

# Confusion Matrix 시각화 (파란색)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'}, annot_kws={"size": 14})
plt.title('Confusion Matrix (XGBoost) - Tear Uric Acid Classification', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('Actual Label', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Feature Importance 시각화 (파란색)
plt.figure(figsize=(8, 6))
sns.barplot(x='Normalized Importance', y='Feature', data=importance_df, color='blue')
plt.title('Feature Importance (XGBoost)', fontsize=16)
plt.xlabel('Normalized Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()