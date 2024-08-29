import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('datasets/my_paypal_creditcard.csv')

X = data.drop(['Class'], axis=1)
y = data['Class']

# Top 10 Negative and Positive Correlations
top_negative_corr = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V18', 'V1', 'V5']
top_positive_corr = ['V11', 'V4', 'V2', 'V21', 'V19', 'V20', 'V23', 'Amount', 'V27', 'V28']

# Combine the lists
top_corr_features = top_negative_corr + top_positive_corr

# Filter X to keep only columns in the top correlation features
X_filtered = X[top_corr_features]

# Print the resulting filtered dataframe columns
print("Columns in X after filtering:")
print(X_filtered.columns)


# Split Test and Train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale X sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
# Solvers: 'liblinear', 'lbfgs', or 'saga'
model = LogisticRegression(
    random_state=42, 
    max_iter=300, 
    class_weight='balanced',
    solver='liblinear',
    verbose=1
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)