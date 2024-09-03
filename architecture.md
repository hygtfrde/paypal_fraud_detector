# V1:
## Split dataset into `removed_dups` and `dups_remain`
- 
  - `removed_dups`
  - `dups_remain`

# V2:
## Focus first on the top 10 correlations

## Top 10 Negative Correlations
| Feature | Correlation |
|---------|-------------|
| V17     | -0.326      |
| V14     | -0.303      |
| V12     | -0.261      |
| V10     | -0.217      |
| V16     | -0.197      |
| V3      | -0.193      |
| V7      | -0.187      |
| V18     | -0.111      |
| V1      | -0.101      |
| V5      | -0.095      |

## Top 10 Positive Correlations
| Feature | Correlation |
|---------|-------------|
| V11     | 0.155       |
| V4      | 0.133       |
| V2      | 0.091       |
| V21     | 0.040       |
| V19     | 0.035       |
| V20     | 0.020       |
| V23     | 0.003       |
| Amount  | 0.006       |
| V27     | 0.018       |
| V28     | 0.010       |


# V3:
## Time-based patterns 
- Normalization (Basic)
    -- ensure that normalization respects time-based dependencies (e.g., rolling windows)
- Binning
    -- different periods (e.g., morning, afternoon, evening) or bin 'Amount' into discrete ranges
- Threshold limits
    -- Use a very high (99%) threshold

# V4:
## Time-based patterns && advanced normalization
- Normalization (Advanced)
    -- advanced normalization techniques (e.g., min-max normalization, log transformations, or z-score normalization)
    -- ensure that normalization respects time-based dependencies (e.g., rolling windows)
- Binning
    -- different periods (e.g., morning, afternoon, evening) or bin 'Amount' into discrete ranges
- Threshold limits
    -- Use a very high (99%) threshold and adjust it based on 'precision-recall'
- Use a Different Metric (e.g., F1-Score or Precision-Recall Curve)
    -- instead of focusing on accuracy, optimize for a balance between precision and recall using the F1-score
    -- plot a precision-recall curve to find the optimal threshold for your model.


# V5
## Class Imbalance (Oversampling and Undersampling)
- SMOTE
- ADASYN
- Ensemble methods

```
# First, apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Then, apply ADASYN to the SMOTE-resampled data
adasyn = ADASYN(random_state=42)
X_train_combined, y_train_combined = adasyn.fit_resample(X_train_smote, y_train_smote)
```

## Voting Classifier
- Combining Logistic Regression and other models into a single Ensemble Method:
```
log_reg = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
gb_model = GradientBoostingClassifier(random_state=42)

# Combine them using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft'  # Use 'soft' for probability voting
)
```

## Stacking
```
estimators = [
    ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
    ('gb', GradientBoostingClassifier(random_state=42))
]

# Stacking model with logistic regression as meta-learner
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42)
)

# Fit the model
stacking_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

## Blending ......

# V6 ?

# Alternative Metrics
- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision Recall AUC

# Feature Engineering
- T-SNE
- UMAP
- Outlier detection
- Interaction times

# Post-modling
- SHAP
- LIME