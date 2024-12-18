import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# File path setup
file_path = 'D:/논문/Uric acid SCL 하영/Figure python/uric acid_10min.xlsx'

# Load data
data = pd.read_excel(file_path)

# Data preprocessing
# Select relevant columns and remove missing values
data = data[['subject_id', 'label', 'blood_uric_acid']].dropna()

# Filter dataset to include only 'Healthy' (label 0) and 'Gout w/o treatment' (label 1)
data = data[data['label'] != 2]  # Exclude 'Gout w/ treatment'

# Separate features and labels
features = data[['blood_uric_acid']]
labels = data['label']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Define label names
label_names = ['Healthy', 'Gout w/o treatment']

# Calculate and print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("Confusion Matrix (Percentage by Row):\n", cm_percentage)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

# Visualize the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Reds', xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Percentage'}, annot_kws={"size": 16})
plt.title('Confusion Matrix (Random Forest) - Healthy vs Gout w/o Treatment', fontsize=22)
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('Actual Label', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Calculate and display feature importance
importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Normalize importance to sum to 1
importance_df['Normalized Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

# Feature importance visualization
plt.figure(figsize=(12, 8))
sns.barplot(x='Normalized Importance', y='Feature', data=importance_df, color='red')
plt.title('Feature Importance (Random Forest)', fontsize=22)
plt.xlabel('Normalized Importance', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Annotate bars with importance values
for index, value in enumerate(importance_df['Normalized Importance']):
    plt.text(value, index, f'{value:.2f}', va='center', fontsize=14)

plt.show()

# Print Feature Importance for Debugging
print("\nFeature Importance (Sorted):")
print(importance_df)
