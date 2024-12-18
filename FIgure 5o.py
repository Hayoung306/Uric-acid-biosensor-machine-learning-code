import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 경로 설정
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'

# 데이터 로드
data = pd.read_excel(file_path)

# 데이터 전처리
# 피처로 사용할 컬럼들: tear_difference, tear_rate, tear_AUC, lag_time, blood_uric_acid, tear_uric_acid
data = data[['subject_id', 'label', 'tear_difference', 'tear_rate', 'tear_AUC', 'lag_time', 'blood_uric_acid', 'tear_uric_acid']]
data = data.dropna()  # 결측값 제거

# 특징과 레이블 분리
features = data[['tear_difference', 'tear_rate', 'tear_AUC', 'lag_time', 'blood_uric_acid', 'tear_uric_acid']]
labels = data['label']

# 데이터 분할 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# XGBoost DMatrix 포맷으로 변환
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# XGBoost 모델 매개변수 정의
params = {
    'objective': 'multi:softmax',  # 다중 클래스 분류
    'num_class': len(labels.unique()),  # 클래스 수 자동 계산
    'learning_rate': 0.1,           # 학습률
    'max_depth': 5,                 # 트리 깊이
    'alpha': 10                     # L1 정규화 항
}

# XGBoost 모델 학습
xgboost_model = xgb.train(params, train_data, num_boost_round=100)

# 예측
y_pred = xgboost_model.predict(test_data)

# Confusion Matrix의 레이블 명 설정
label_names = ['Healthy', 'Gout w/o treatment', 'Gout w/ treatment']

# 성능 평가 및 Confusion Matrix 계산
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix를 각 행 기준으로 100분율로 변환
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Confusion Matrix 및 Classification Report 출력
print("Confusion Matrix (행 기준 백분율):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

# Confusion Matrix 시각화 (백분율 표시)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix (XGBoost) - Percentage per Class')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
