import pickle
import shap
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import threading

# Assuming you have the scaler, X_train_scaled, X_test_scaled, y_test, and X.columns available
# Make sure these are loaded or regenerated from your original dataset

# Function to automatically close plots after a delay
def close_plot_after_delay(fig, delay):
    def close():
        plt.close(fig)
    timer = threading.Timer(delay, close)
    timer.start()
    
# Load the original dataset
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

# Split data into features and labels
X = data.drop(['Class'], axis=1)
y = data['Class']

# Reproduce the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE and ADASYN
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

adasyn = ADASYN(random_state=42)
X_train_combined, y_train_combined = adasyn.fit_resample(X_train_smote, y_train_smote)

# Apply scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# Load the pickled model
with open('voting_model.pkl', 'rb') as model_file:
    voting_model = pickle.load(model_file)

# Ensure you have X_train_scaled, X_test_scaled, y_test, and feature names (X.columns)
# For example:
# X_train_scaled = ...
# X_test_scaled = ...
# y_test = ...
# feature_names = X.columns

# Extract the individual models from the VotingClassifier
rf_model = voting_model.named_estimators_['rf']
gb_model = voting_model.named_estimators_['gb']
adaboost_model = voting_model.named_estimators_['adaboost']

# --------------------------------------------
# SHAP for RandomForest
if hasattr(rf_model, "estimators_"):  # Ensure model is trained
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(X_test_scaled)
    shap.summary_plot(rf_shap_values[1], X_test_scaled, feature_names=X.columns, show=False)  # Class 1 SHAP values
    plt.title("SHAP Summary Plot for RandomForest")
    close_plot_after_delay(plt.gcf(), 30)
else:
    print("RandomForest model is not fitted or does not have estimators_ attribute.")

# SHAP for GradientBoosting
if hasattr(gb_model, "estimators_"):  # Ensure model is trained
    gb_explainer = shap.TreeExplainer(gb_model)
    gb_shap_values = gb_explainer.shap_values(X_test_scaled)
    shap.summary_plot(gb_shap_values[1], X_test_scaled, feature_names=X.columns, show=False)  # Class 1 SHAP values
    plt.title("SHAP Summary Plot for GradientBoosting")
    close_plot_after_delay(plt.gcf(), 30)
else:
    print("GradientBoosting model is not fitted or does not have estimators_ attribute.")

# SHAP for AdaBoost using KernelExplainer
# KernelExplainer is computationally heavy; use a subset of X_train_scaled for efficiency
ada_explainer = shap.KernelExplainer(adaboost_model.predict_proba, X_train_scaled[:100])  
ada_shap_values = ada_explainer.shap_values(X_test_scaled[:100])  # Limit to first 100 samples for speed
shap.summary_plot(ada_shap_values[1], X_test_scaled[:100], feature_names=X.columns, show=False)  # Class 1 SHAP values
plt.title("SHAP Summary Plot for AdaBoost")
close_plot_after_delay(plt.gcf(), 30)

# --------------------------------------------
# LIME Explanations

# LIME for RandomForest
lime_explainer_rf = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Not Fraud', 'Fraud'],
    discretize_continuous=True
)
lime_instance_idx = np.random.randint(0, X_test_scaled.shape[0])  # Randomly choose an instance
lime_explanation_rf = lime_explainer_rf.explain_instance(X_test_scaled[lime_instance_idx], rf_model.predict_proba)
lime_explanation_rf.show_in_notebook(show_all=False)

# LIME for GradientBoosting
lime_explainer_gb = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Not Fraud', 'Fraud'],
    discretize_continuous=True
)
lime_explanation_gb = lime_explainer_gb.explain_instance(X_test_scaled[lime_instance_idx], gb_model.predict_proba)
lime_explanation_gb.show_in_notebook(show_all=False)

# LIME for AdaBoost
lime_explainer_ada = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Not Fraud', 'Fraud'],
    discretize_continuous=True
)
lime_explanation_ada = lime_explainer_ada.explain_instance(X_test_scaled[lime_instance_idx], adaboost_model.predict_proba)
lime_explanation_ada.show_in_notebook(show_all=False)
# --------------------------------------------
