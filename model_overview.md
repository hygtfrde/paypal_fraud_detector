# Credit Card Fraud Detection Model Explanation

## Overview
This model is designed to detect fraudulent transactions using a credit card dataset. It utilizes a **Voting Classifier** that combines three powerful machine learning algorithms: **Random Forest**, **AdaBoost**, and **Gradient Boosting**. The model performs oversampling to handle class imbalance, followed by scaling for normalization, and then it trains and evaluates the ensemble model.

## Workflow

### 1. **Data Preparation**
   - **Dataset**: The model uses a credit card transaction dataset where the target variable is 'Class'. 
     - `X`: All features except the 'Class' column.
     - `y`: The 'Class' column (1: Fraud, 0: Not Fraud).
   - The dataset is split into training and testing sets using an 80-20 split.

### 2. **Handling Class Imbalance**
   - **SMOTE (Synthetic Minority Over-sampling Technique)**: SMOTE generates synthetic samples for the minority class (fraud cases) to balance the training set.
   - **ADASYN (Adaptive Synthetic Sampling)**: ADASYN is applied to the SMOTE-resampled data, further balancing the dataset by generating more challenging samples from the minority class.
   
### 3. **Data Scaling**
   - **MinMaxScaler**: All features are scaled to a range of 0 to 1 to ensure uniformity and improve model performance. This step is applied to both the training and testing sets.

### 4. **Model Setup**
   - The ensemble model is created using a **Voting Classifier** with a 'soft' voting strategy, meaning it averages the predicted probabilities from the individual classifiers.
   - **Base Models**:
     - **RandomForestClassifier**: A bagging technique that uses a collection of decision trees to make predictions.
     - **AdaBoostClassifier**: A boosting algorithm that builds multiple weak learners (decision trees) sequentially, each correcting the errors of the previous one.
     - **GradientBoostingClassifier**: Another boosting technique that combines weak learners to create a strong predictive model.
   - Each of these base models is configured with 300 estimators to ensure sufficient model complexity.

### 5. **Model Training**
   - The training process is timed and output is made verbose for better insight into the progress.
   - The model is trained on the scaled, oversampled dataset.

### 6. **Model Evaluation**
   - **AUPRC (Area Under the Precision-Recall Curve)**: The average precision score is computed and reported. This metric is particularly useful in imbalanced datasets, as it focuses on the model's performance in distinguishing the minority class (fraud cases).
   - **Precision-Recall Curve**: The precision-recall curve is plotted to visualize the trade-off between precision and recall at various threshold levels.

### 7. **Threshold Selection**
   - The optimal threshold is determined by analyzing the precision-recall curve, aiming to balance precision and recall effectively.
   - A fixed threshold (`0.65`) is then applied to the predicted probabilities to make binary predictions (fraud vs. not fraud).

### 8. **Performance Reporting**
   - **Classification Report**: This report includes precision, recall, F1-score, and accuracy metrics for the final model predictions.
   - **Confusion Matrix**: The confusion matrix provides a detailed breakdown of the model’s predictions, showing the number of true positives, true negatives, false positives, and false negatives.

### 9. **Feature Importance**
   - Feature importance values are extracted from the **Random Forest** and **Gradient Boosting** models, which help in identifying the most influential features in predicting fraudulent transactions. These values are then merged and reported.

### 10. **Visualization**
   - The model uses **matplotlib** to visualize the precision-recall curve. To ensure smooth execution in non-interactive environments, the plot is automatically closed after a 30-second delay using a threaded timer.

## Summary
The model leverages an ensemble of classifiers to detect fraudulent credit card transactions in an imbalanced dataset. By combining the strengths of multiple algorithms and carefully handling class imbalance, the model achieves high precision and recall, making it an effective tool for fraud detection.
