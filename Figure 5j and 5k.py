import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'
data = pd.read_excel(file_path)

# Data preprocessing
data = data[['label', 'blood_uric_acid', 'tear_uric_acid']].dropna()

# Only include Gout w/o treatment (1) and Gout w/ treatment (2)
data = data[data['label'] != 0]
data['label'] = data['label'].replace({1: 0, 2: 1})  # Relabel to 0 and 1

# Separate features and labels
features = data[['blood_uric_acid', 'tear_uric_acid']]
labels = data['label']

# Standardize features
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Convert to XGBoost DMatrix
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost model parameters
params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 5,
    'colsample_bytree': 0.8
}

# Train the model
xgboost_model = xgb.train(params, train_data, num_boost_round=100)

# Make predictions
y_pred = xgboost_model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)

# Confusion matrix
label_names = ['Gout w/o treatment', 'Gout w/ treatment']
cm = confusion_matrix(y_test, y_pred_binary)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix in purple color scheme
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Purples', xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'}, annot_kws={"size": 14})
plt.title('Confusion Matrix (XGBoost) - Combined Features', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('Actual Label', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Feature importance
importance_dict = xgboost_model.get_score(importance_type='weight')
importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
importance_df['Normalized Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

# Plot feature importance in purple
plt.figure(figsize=(8, 6))
sns.barplot(x='Normalized Importance', y='Feature', data=importance_df, color='purple')
plt.title('Feature Importance (XGBoost)', fontsize=16)
plt.xlabel('Normalized Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
