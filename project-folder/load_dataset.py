"""
Load and prepare the Breast Cancer Wisconsin (Diagnostic) Dataset
Dataset Source: UCI Machine Learning Repository / sklearn.datasets
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load the dataset
print("Loading Breast Cancer Wisconsin (Diagnostic) Dataset...")
cancer = load_breast_cancer()

# Create DataFrame
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.DataFrame(cancer.target, columns=['diagnosis'])

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save to CSV
df.to_csv('data/breast_cancer_dataset.csv', index=False)
print(f"Dataset saved to data/breast_cancer_dataset.csv")

# Print dataset information
print(f"\nDataset Information:")
print(f"- Total Instances: {len(df)}")
print(f"- Total Features: {len(cancer.feature_names)}")
print(f"- Target Classes: {np.unique(y.values)}")
print(f"- Class Distribution:")
print(y['diagnosis'].value_counts())
print(f"\nFeatures ({len(cancer.feature_names)}):")
for i, feature in enumerate(cancer.feature_names, 1):
    print(f"  {i}. {feature}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save split datasets
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print(f"\nTrain-Test Split (80-20):")
print(f"- Training Instances: {len(X_train)}")
print(f"- Testing Instances: {len(X_test)}")
print("\nSplit datasets saved to data/ directory")
