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
1. **Feature Engineering (T-SNE, UMAP):**  
   - Techniques like T-SNE, UMAP, or advanced feature engineering are not implemented.

2. **Stacking:**  
   - The model does not use stacking techniques; only voting (ensemble) methods are implemented.

3. **Outlier Detection:**  
   - No explicit outlier detection techniques are implemented in the code.

4. **Post-Modeling (SHAP, LIME, Pickle, Dumps):**  
   - Post-modeling interpretation tools like SHAP or LIME, and model persistence (e.g., Pickle) are not included.
