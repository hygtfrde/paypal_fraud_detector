import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier, 
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
import seaborn as sns
import matplotlib.pyplot as plt

"""
- This model takes roughly 7 hours to train on a 16 GB machine
- It is used more for educational purposes than for research, deployment, etc.
"""


# --------------------------------------------
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

X = data.drop(['Class'], axis=1)
y = data['Class']
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
# Create individual models
rf_model = RandomForestClassifier(random_state=42, n_estimators=300, class_weight='balanced', verbose=1)
adaboost_model = AdaBoostClassifier(random_state=42, n_estimators=300)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=300)

# Define the stacking classifier with base learners and a meta-learner
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('adaboost', adaboost_model),
        ('gb', gb_model)
    ],
    final_estimator=LogisticRegression(random_state=42)
)

# Fit the stacking model
stacking_model.fit(X_train_scaled, y_train_combined)

# Make predictions with stacking
y_pred_stacking_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]


# --------------------------------------------
# Implement blending using cross-validation to avoid data leakage
skf = StratifiedKFold(n_splits=5)
blend_train = np.zeros((X_train_scaled.shape[0], 3))
blend_test = np.zeros((X_test_scaled.shape[0], 3))

for train_idx, holdout_idx in skf.split(X_train_scaled, y_train_combined):
    X_tr, X_ho = X_train_scaled[train_idx], X_train_scaled[holdout_idx]
    y_tr, y_ho = y_train_combined[train_idx], y_train_combined[holdout_idx]

    # Fit base models
    rf_model.fit(X_tr, y_tr)
    adaboost_model.fit(X_tr, y_tr)
    gb_model.fit(X_tr, y_tr)

    # Get holdout set predictions
    blend_train[holdout_idx, 0] = rf_model.predict_proba(X_ho)[:, 1]
    blend_train[holdout_idx, 1] = adaboost_model.predict_proba(X_ho)[:, 1]
    blend_train[holdout_idx, 2] = gb_model.predict_proba(X_ho)[:, 1]

    # Accumulate test set predictions
    blend_test[:, 0] += rf_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits
    blend_test[:, 1] += adaboost_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits
    blend_test[:, 2] += gb_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

# Train meta-learner on blended features
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(blend_train, y_train_combined)
y_pred_blend = meta_learner.predict(blend_test)


# --------------------------------------------
# Create Voting Classifier with stacking and blending included
# Note: Voting Classifier should be created based on individual models' outputs.
# Stacking and blending are typically used separately or as a final step, not combined directly in VotingClassifier.
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('adaboost', adaboost_model),
        ('gb', gb_model)
    ],
    voting='soft'
)

# Fit the voting model on the original scaled training data
voting_model.fit(X_train_scaled, y_train_combined)

# Evaluate predictions from the combined voting model
y_pred_combined = voting_model.predict(X_test_scaled)

# --------------------------------------------
# Plot the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_stacking_proba)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Find the optimal threshold based on precision-recall curve
optimal_idx = np.argmax(precision - recall)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]


# --------------------------------------------
# Make predictions with the optimal threshold
threshold = 0.60
y_pred_stacking = (y_pred_stacking_proba >= threshold).astype(int)
y_pred_combined_threshold = (voting_model.predict_proba(X_test_scaled)[:, 1] >= threshold).astype(int)

print("\nClassification Report for Stacking Model:")
print(classification_report(y_test, y_pred_stacking))

print("\nConfusion Matrix for Stacking Model:")
print(confusion_matrix(y_test, y_pred_stacking))

print("\nClassification Report for Blending Model:")
print(classification_report(y_test, y_pred_blend))

print("\nConfusion Matrix for Blending Model:")
print(confusion_matrix(y_test, y_pred_blend))

print("\nClassification Report for Voting Model:")
print(classification_report(y_test, y_pred_combined_threshold))

print("\nConfusion Matrix for Voting Model:")
print(confusion_matrix(y_test, y_pred_combined_threshold))

print(f"\nOptimal Threshold: {optimal_threshold}")

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
