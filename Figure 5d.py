import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# File path setup
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'

# Load data
data = pd.read_excel(file_path)

# Data preprocessing
# Select relevant columns
data = data[['subject_id', 'label', 'blood_uric_acid']]
data = data.dropna()  # Remove missing values

# Filter dataset to include only 'Healthy' (label 0) and 'Gout w/o treatment' (label 1)
data = data[data['label'] != 2]  # Exclude 'Gout w/ treatment'

# Separate features and labels
features = data[['blood_uric_acid']]
labels = data['label']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert data into XGBoost DMatrix format
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost model parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'learning_rate': 0.1,           # Learning rate
    'max_depth': 5,                 # Maximum tree depth
    'alpha': 10                     # L1 regularization term
}

# Train the XGBoost model
xgboost_model = xgb.train(params, train_data, num_boost_round=100)

# Make predictions
y_pred = xgboost_model.predict(test_data)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary classes

# Define label names
label_names = ['Healthy', 'Gout w/o treatment']

# Calculate and print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_binary)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("Confusion Matrix (Percentage by Row):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary, target_names=label_names))

# Visualize the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Reds', xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'}, annot_kws={"size": 16})
plt.title('Confusion Matrix (XGBoost) - Healthy vs Gout w/o Treatment', fontsize=22)
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('Actual Label', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
