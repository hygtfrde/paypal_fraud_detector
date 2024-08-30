import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE  # Added for SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

"""
without binning:
 [[56856     8]
 [   18    80]]
 
with binning:
 [[56861     3]
 [   20    78]]
"""

# Load dataset
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

X = data.drop(['Class'], axis=1)
y = data['Class']


# Split Test and Train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale X sets using MinMaxScaler for advanced normalization
scaler = MinMaxScaler()  # Changed from StandardScaler to MinMaxScaler
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)



# --------------------------------------------
# Create individual models
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', verbose=1)
adaboost_model = AdaBoostClassifier(random_state=42, n_estimators=100)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)

# Combine them using Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('adaboost', adaboost_model),
        ('gb', gb_model)
    ],
    voting='soft'  # Use 'soft' for averaging predicted probabilities, or 'hard' for majority voting
)

# Fit the ensemble model
ensemble_model.fit(X_train_scaled, y_train_resampled)

# Make predictions
y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
# --------------------------------------------



# --------------------------------------------
# Plot the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Find the optimal threshold based on precision-recall curve
optimal_idx = np.argmax(precision - recall)

# Ensure the optimal index does not exceed the size of the thresholds array
if optimal_idx >= len(thresholds):
    optimal_idx = len(thresholds) - 1

optimal_threshold = thresholds[optimal_idx]


# --------------------------------------------
# Make predictions with the optimal threshold
# y_pred_threshold = (y_pred_proba >= optimal_threshold).astype(int)

threshold = 0.69

# Make predictions with the hardcoded threshold
y_pred_threshold = (y_pred_proba >= threshold).astype(int)

print("\nClassification Report with optimal threshold:")
print(classification_report(y_test, y_pred_threshold))

print("\nConfusion Matrix with optimal threshold:")
print(confusion_matrix(y_test, y_pred_threshold))

# --------------------------------------------
# Extract feature importances from individual models
try:
    feature_importance_rf = rf_model.feature_importances_
    feature_importance_gb = gb_model.feature_importances_

    # Create dataframes for feature importances from both models
    feature_importance_rf_df = pd.DataFrame({'feature': X.columns, 'importance_rf': feature_importance_rf})
    feature_importance_gb_df = pd.DataFrame({'feature': X.columns, 'importance_gb': feature_importance_gb})

    # Merge or compare the feature importances
    feature_importance_combined = feature_importance_rf_df.merge(feature_importance_gb_df, on='feature')
    print("\nCombined Feature Importances (from RandomForest and GradientBoosting):")
    print(feature_importance_combined)

except AttributeError:
    print("Some models do not support feature importances, skipping feature importance extraction.")

# Print optimal threshold
print(f"\nOptimal Threshold: {optimal_threshold}")
