import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('datasets/my_paypal_creditcard.csv')

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Plot 1: Distribution of the 'Class' column (fraud vs non-fraud)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', hue='Class', data=data, palette='viridis', legend=False)
plt.title('Distribution of Class (Fraud vs Non-Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('visualizations/class_distribution.png')
plt.close()

# Plot 2: Correlation heatmap of features
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# Plot 3: Distribution of 'Amount' column
plt.figure(figsize=(6, 4))
sns.histplot(data['Amount'], kde=True, bins=40, color='blue')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig('visualizations/amount_distribution.png')
plt.close()

# Plot 4: Distribution of 'Time' column
plt.figure(figsize=(6, 4))
sns.histplot(data['Time'], kde=True, bins=40, color='green')
plt.title('Transaction Time Distribution')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.savefig('visualizations/time_distribution.png')
plt.close()

# Plot 5: Boxplot of Amount by Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Amount', hue='Class', data=data, palette='muted', legend=False)
plt.title('Boxplot of Amount by Class')
plt.xlabel('Class')
plt.ylabel('Transaction Amount')
plt.savefig('visualizations/amount_by_class.png')
plt.close()
