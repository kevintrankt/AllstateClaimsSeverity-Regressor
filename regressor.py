# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

import pandas

# Training Data
dataset = pandas.read_csv('/Users/kevin/Documents/AllstateData/train.csv')
# Testing Data
dataset_test = pandas.read_csv("/Users/kevin/Documents/AllstateData/test.csv")

# Save the id's for submission file
ID = dataset_test['id']
# Drop id column
dataset_test.drop('id', axis=1, inplace=True)

# Print first 5 rows of each column in training data
# print(dataset.head(5))

print(dataset.shape)