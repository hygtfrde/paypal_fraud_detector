import pandas as pd

# Load the dataset
data = pd.read_csv('datasets/my_paypal_creditcard.csv')

# Basic DataFrame Information
print("DataFrame Information:")
print("----------------------")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")
print(f"DataFrame shape: {data.shape}")
print("\n")

# Check for missing values
print("Missing Values per Column:")
print("--------------------------")
print(data.isnull().sum())
print("\n")

# Display data types of each column
print("Data Types of Each Column:")
print("--------------------------")
print(data.dtypes)
print("\n")

# Display first few rows to get a sense of the data
print("First 5 Rows of the DataFrame:")
print("------------------------------")
print(data.head())
print("\n")

# Basic statistics of the numeric columns
print("Basic Statistics of Numeric Columns:")
print("-----------------------------------")
print(data.describe())
print("\n")

# Check for class imbalance
if 'Class' in data.columns:
    print("Class Distribution:")
    print("-------------------")
    print(data['Class'].value_counts())
else:
    print("Column 'Class' not found in the dataset.")
