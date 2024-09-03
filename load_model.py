import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the pickled model
with open('voting_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load and preprocess new data
# Alt: create new data to make predictions and skip csv or df loading
# Assuming new_data is a DataFrame similar to the original training data
new_data = pd.read_csv('path_to_new_data.csv')

# Ensure new data is preprocessed in the same way as training data
# Example: Scaling the data
scaler = MinMaxScaler()
new_data_scaled = scaler.fit_transform(new_data)

# Predict with the loaded model
y_pred_proba = loaded_model.predict_proba(new_data_scaled)[:, 1]

# Example: Thresholding and reporting
threshold = 0.65  # or the threshold you determined previously
y_pred = (y_pred_proba >= threshold).astype(int)

print("\nPredictions:")
print(y_pred)
