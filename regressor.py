import warnings
import pandas

# Ignore warnings
warnings.filterwarnings('ignore')

# Training Data
dataset = pandas.read_csv('/Users/kevin/Documents/AllstateData/train.csv')
# Testing Data
dataset_test = pandas.read_csv("/Users/kevin/Documents/AllstateData/test.csv")

ID = dataset_test['id']
# Drop id column
dataset_test.drop('id', axis=1, inplace=True)
# Drop id column
dataset = dataset.iloc[:, 1:]

# Print first 5 rows of each column in training data
print('\nFirst five rows of dataset:', dataset.head(5))
print('\nDataset Shape: ', dataset.shape)
print('\nTesting Dataset Shape:', dataset_test.shape)
print('\nDataset Description:', dataset.describe())
print('\nDataset Skew', dataset.skew())


# Visualize Datasets (Violin Plots)
