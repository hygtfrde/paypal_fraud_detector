import pandas as pd
import numpy as np


df = pd.read_csv('datasets/my_paypal_creditcard.csv')

print("Number of rows and columns:", df.shape)
print("\nData Types:\n", df.dtypes)

print("\nSummary Statistics:\n", df.describe())

print("Additional important information")
print("\nNull Values in Each Column:\n", df.isnull().sum())
print("\nMean Values:\n", df.mean())
print("\nMinimum Values:\n", df.min())
print("\nMaximum Values:\n", df.max())

print("\nNumber of Duplicate Rows:", df.duplicated().sum())

print("\nCorrelation Matrix:\n", df.corr())