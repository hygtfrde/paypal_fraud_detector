# Welcome to My Paypal
***

## Task
Analyze a dataset of credit card transaction where the vast majority are non-fraudulent and there is a small class of fraudulent ones, specifically we have 492 frauds out of 284,807 transactions. Then build an ML model to predict the fraudulent transactions. 

## Description
To build this model, start by loading and preprocessing the data. The dataset is split into training and test sets, followed by handling the class imbalance using techniques like SMOTE and ADASYN for oversampling. MinMaxScaler is then applied to normalize the feature set. This ensures that all the features are scaled within a similar range, which improves the performance of the models.

Next, multiple classifiers (RandomForest, AdaBoost, and GradientBoosting) are trained as base models. These models are combined using a soft-voting ensemble method, where each model's predicted probabilities are averaged to make the final prediction. The model's performance is evaluated by measuring metrics like Average Precision (AUPRC) and generating a Precision-Recall curve. For further fine-tuning, an optimal threshold for classification is determined from the precision-recall curve.

Finally, the model’s predictions are assessed with classification reports and confusion matrices, and feature importances are extracted from the tree-based models (RandomForest and GradientBoosting). The combination of models and advanced sampling techniques leads to a robust solution for detecting credit card fraud.


## Installation
Install the dependencies with Pip:
```
pip install -r requirements.txt
```

## Usage
Open the Jupyter Notebook and run the cells. 
Or to run the Python code directly run this command:
```
python models/multi_var_log_reg_model_v8.py
```
You can replace the Version number (v1, v2, etc.) to test different types of the model.

### The Core Team


<span><i>Made at <a href="https://qwasar.io">Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt="Qwasar SV -- Software Engineering School's Logo" src="https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png" width="20px" /></span>