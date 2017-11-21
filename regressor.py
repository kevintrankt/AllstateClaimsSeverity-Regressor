import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings('ignore')

# Training Data
dataset = pd.read_csv('/Users/kevin/Documents/AllstateData/train.csv')
# Testing Data
dataset_test = pd.read_csv("/Users/kevin/Documents/AllstateData/test.csv")

ID = dataset_test['id']
# Drop id column
dataset_test.drop('id', axis=1, inplace=True)
# Drop id column
dataset = dataset.iloc[:, 1:]

# Print first 5 rows of each column in training data
# print('\nFirst five rows of dataset:', dataset.head(5))
# print('\nDataset Shape: ', dataset.shape)
# print('\nTesting Dataset Shape:', dataset_test.shape)
# print('\nDataset Description:', dataset.describe())
# print('\nDataset Skew', dataset.skew())

# Visualize Datasets (Violin Plots)
split = 116  # range of features considered
size = 15  # number of features considered

data = dataset.iloc[:, split:]  # dataframe with only continuous features
cols = data.columns  # column names
print('Column Names', cols)

# Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7
for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i * n_cols + j], data=dataset, ax=ax[j])
# Display plot
plt.show()
