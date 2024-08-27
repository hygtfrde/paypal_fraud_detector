import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('datasets/my_paypal_creditcard.csv')

# Basic Info
print("Number of rows and columns:", df.shape)
print("\nData Types:\n", df.dtypes)

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Additional important information
print("\nNull Values in Each Column:\n", df.isnull().sum())
print("\nMean Values:\n", df.mean())
print("\nMinimum Values:\n", df.min())
print("\nMaximum Values:\n", df.max())

# Checking for duplicate rows
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Checking for any correlations
print("\nCorrelation Matrix:\n", df.corr())