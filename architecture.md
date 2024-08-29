# V1:
## Split dataset into `removed_dups` and `dups_remain`
    |
    |___> removed_dups
    |___> dups_remain 

# V2:
## Focus first on the top 10 corrs
### Top 10 Negative Correlations:
- V17: Correlation = -0.326
- V14: Correlation = -0.303
- V12: Correlation = -0.261
- V10: Correlation = -0.217
- V16: Correlation = -0.197
- V3: Correlation = -0.193
- V7: Correlation = -0.187
- V18: Correlation = -0.111
- V1: Correlation = -0.101
- V5: Correlation = -0.095
### Top 10 Positive Correlations:
- V11: Correlation = 0.155
- V4: Correlation = 0.133
- V2: Correlation = 0.091
- V21: Correlation = 0.040
- V19: Correlation = 0.035
- V20: Correlation = 0.020
- V23: Correlation = 0.003
- Amount: Correlation = 0.006
- V27: Correlation = 0.018
- V28: Correlation = 0.010

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




# Class Imbalance (Oversampling and Undersampling)
- SMOTE
- ADASYN
- Ensemble methods ... ?

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