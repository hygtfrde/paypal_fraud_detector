import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt

import time
import threading
import pickle


# --------------------------------------------
# Helper Functions
def close_plot_after_delay(fig, delay):
    def close():
        plt.close(fig)
    
    timer = threading.Timer(delay, close)
    timer.start()
# --------------------------------------------


# --------------------------------------------
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

X = data.drop(['Class'], axis=1)
y = data['Class']
# --------------------------------------------


# --------------------------------------------
# Split Test and Train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# First, apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Then, apply ADASYN to the SMOTE-resampled data
adasyn = ADASYN(random_state=42)
X_train_combined, y_train_combined = adasyn.fit_resample(X_train_smote, y_train_smote)

# Scale X sets using MinMaxScaler for advanced normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test)
# --------------------------------------------


# --------------------------------------------
# Create individual models with added verbosity
rf_model = RandomForestClassifier(random_state=42, n_estimators=300, class_weight='balanced', verbose=2)
adaboost_model = AdaBoostClassifier(random_state=42, n_estimators=300)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=300, verbose=2)
# --------------------------------------------


# --------------------------------------------
# Timing and verbose output for training voting model
start_time = time.time()

# Create Voting Classifier with RandomForest, AdaBoost, and GradientBoosting
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('adaboost', adaboost_model),
        ('gb', gb_model)
    ],
    voting='soft'
)

print("Starting training of the voting model...")
voting_model.fit(X_train_scaled, y_train_combined)

end_time = time.time()
training_duration = end_time - start_time

print(f"\nTraining completed in {training_duration // 60} minutes and {training_duration % 60:.2f} seconds.")
# --------------------------------------------


# --------------------------------------------
# Make predictions with the voting classifier
y_pred_combined_proba = voting_model.predict_proba(X_test_scaled)[:, 1]

# Calculate and print AUPRC
auprc_voting = average_precision_score(y_test, y_pred_combined_proba)
print(f"AUPRC for Voting Model: {auprc_voting:.4f}")

# Save AUPRC for reporting or model comparison
with open('model_performance.txt', 'a') as f:
    f.write(f"AUPRC for Voting Model: {auprc_voting:.4f}\n")
# --------------------------------------------


# --------------------------------------------
# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_combined_proba)

# Create the plot
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='.')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')

# Automatically close the plot after 30 seconds
close_plot_after_delay(fig, 30)

# Show the plot
plt.show()
# --------------------------------------------

# Find the optimal threshold based on precision-recall curve
optimal_idx = np.argmax(precision - recall)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
print(f"\nOptimal Threshold: {optimal_threshold}")

# Save optimal threshold for reporting or model comparison
with open('model_performance.txt', 'a') as f:
    f.write(f"Optimal Threshold: {optimal_threshold}\n")
# --------------------------------------------


# --------------------------------------------
# Make predictions with the optimal threshold
threshold = 0.65
y_pred_combined_threshold = (y_pred_combined_proba >= threshold).astype(int)

print("\nClassification Report for Voting Model:")
print(classification_report(y_test, y_pred_combined_threshold))

print("\nConfusion Matrix for Voting Model:")
print(confusion_matrix(y_test, y_pred_combined_threshold))
# --------------------------------------------


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
# --------------------------------------------


# --------------------------------------------
# Pickle Dump
with open('voting_model.pkl', 'wb') as model_file:
    pickle.dump(voting_model, model_file)
# --------------------------------------------