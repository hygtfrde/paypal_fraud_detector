# Analysis of `model_stable_v.py` modified from V8 in `older_models`

## Correctly Implemented:
1. **Class Imbalance (Oversampling and Undersampling):**  
   - Both SMOTE and ADASYN are used to handle class imbalance, as seen in the model pipeline.

2. **Ensemble Methods (Voting Classifier):**  
   - A VotingClassifier is implemented combining RandomForest, AdaBoost, and GradientBoosting models.

3. **Advanced Normalization:**  
   - MinMaxScaler is used for advanced normalization of feature sets.

4. **Alternative Metrics (Precision-Recall AUC):**  
   - The model calculates and reports AUPRC (Area Under the Precision-Recall Curve).

5. **Precision-Recall Curve & Optimal Threshold:**  
   - Precision-recall curves are computed, and the model identifies an optimal threshold based on the curve.

## Not Implemented:
0. **Duplicate Removal:**
   - Removing duplicates from the data set may result in removing precious fraudulent data points, so it is not used.

1. **Feature Engineering (T-SNE, UMAP):**  
   - Techniques like T-SNE, UMAP, or advanced feature engineering are not implemented.
   - To gain deeper insights into clusters or non-linear relationships within the data, T-SNE or UMAP could still offer value

2. **Stacking:**  
   - The model does not use stacking techniques; only voting (ensemble) methods are implemented.
   - Stacking (as well as Blending) appears to increase training times by an unmanageable amount (about 7 hours total), so it is not used.

3. **Outlier Detection:**  
   - No explicit outlier detection techniques are implemented in the code.
   - If PCA effectively captured the essential variance and smoothed the effects of outliers by projecting the data into a lower-dimensional space, further outlier detection may provide diminishing returns.

4. **Post-Modeling (SHAP, LIME, Pickle, Dumps):**  
   - Post-modeling interpretation tools like SHAP or LIME, and model persistence (e.g., Pickle) are not included.


# Performance Output:
```
AUPRC for Voting Model: 0.8811

Optimal Threshold: 0.8424429208544737

Classification Report for Voting Model:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.88      0.85      0.86        98

    accuracy                           1.00     56962
   macro avg       0.94      0.92      0.93     56962
weighted avg       1.00      1.00      1.00     56962


Confusion Matrix for Voting Model:
[[56853    11]
 [   15    83]]
```