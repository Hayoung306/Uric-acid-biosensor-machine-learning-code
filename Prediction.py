import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load Data
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'
data = pd.read_excel(file_path)

# Step 2: Define Features and Target
features = data[['tear_uric_acid', 'lag_time', 'tear_uric_acid_real_time']]
target = data['blood_uric_acid']

# Step 3: Standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Step 4: Split the standardized data into training and test sets (80% train, 20% test)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Step 5: Convert data to XGBoost DMatrix format
train_data = xgb.DMatrix(X_train_scaled, label=y_train)
test_data = xgb.DMatrix(X_test_scaled, label=y_test)

# Step 6: Define the XGBoost model parameters`
params = {
    'objective': 'reg:squarederror',  # 회귀 모델을 위한 손실 함수 (평균 제곱 오차)
    'learning_rate': 0.1,             # 학습률
    'max_depth': 5,                   # 최대 트리 깊이
    'alpha': 10                       # 정규화 항
}

# Step 7: Train the model
xgboost_model = xgb.train(params, train_data, num_boost_round=100)

# Step 8: Make predictions on the test set
y_pred_xgb = xgboost_model.predict(test_data)
import numpy as np

# Step 9: Evaluate the model's performance (Updated to calculate r)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
r_xgb = np.sqrt(r2_xgb) if r2_xgb >= 0 else -np.sqrt(-r2_xgb)

print(f"XGBoost Model - Mean Squared Error: {mse_xgb}")
print(f"XGBoost Model - R² Score: {r2_xgb}")
print(f"XGBoost Model - Correlation Coefficient (r): {r_xgb}")

# Step 10: Plot the actual vs predicted values with r displayed
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Blood Uric Acid')
plt.ylabel('Predicted Blood Uric Acid')
plt.title('XGBoost Model: Actual vs Predicted')

# Adding r value as text annotation
plt.text(x=min(y_test), y=max(y_pred_xgb), s=f"r = {r_xgb:.3f}", fontsize=12, color='green')

plt.legend()
plt.show()
