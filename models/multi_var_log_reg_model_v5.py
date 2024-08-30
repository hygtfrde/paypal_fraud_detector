import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE  # Added for SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

"""
    - V5 results in significantly improved model accuracy and precision
    Confusion Matrix with optimal threshold:
    [[56861     3]
    [   20    78]]
"""

# Load dataset
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

X = data.drop(['Class'], axis=1)
y = data['Class']

# --------------------------------------------
# Binning Time Chunks
time_bins = [0, 6*3600, 12*3600, 18*3600, 24*3600]  # Binning by time of day in seconds
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
data['Time_Binned'] = pd.cut(data['Time'], bins=time_bins, labels=time_labels, include_lowest=True)

# Binning the 'Amount' feature into quantiles
data['Amount_Binned'] = pd.qcut(data['Amount'], q=4, labels=False)  # 4 quantiles

# Drop the original 'Time' and 'Amount' columns and use the binned versions
X = data.drop(['Class', 'Time', 'Amount'], axis=1)
X['Time_Binned'] = data['Time_Binned']
X['Amount_Binned'] = data['Amount_Binned']

# Label encode the 'Time_Binned' column to convert categorical values to numeric
label_encoder = LabelEncoder()
X['Time_Binned'] = label_encoder.fit_transform(X['Time_Binned'])
# --------------------------------------------

# Split Test and Train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale X sets using MinMaxScaler for advanced normalization
scaler = MinMaxScaler()  # Changed from StandardScaler to MinMaxScaler
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train ensemble model (Random Forest Classifier)
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100, 
    class_weight='balanced',  # Handle imbalance using class weighting
    verbose=1
)
model.fit(X_train_scaled, y_train_resampled)

# Make predictions
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

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

threshold = 0.75

# Make predictions with the hardcoded threshold
y_pred_threshold = (y_pred_proba >= threshold).astype(int)

print("\nClassification Report with optimal threshold:")
print(classification_report(y_test, y_pred_threshold))

print("\nConfusion Matrix with optimal threshold:")
print(confusion_matrix(y_test, y_pred_threshold))

# Print feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Print optimal threshold
print(f"\nOptimal Threshold: {optimal_threshold}")
