
#Data Loading and Exploration
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Get current working directory
cwd = os.getcwd()


# Construct file path
file_path = os.path.join(cwd, "data\QualityOfService5GDataset.csv")
print("File path:", file_path)

# Check if file exists
if os.path.isfile(file_path):
    print("File exists.")
else:
    print("File does not exist.")

# Load data from CSV file
# Load data with error handling
try:
    # Attempt to load data with default parser
    data = pd.read_csv(file_path)
except pd.errors.ParserError:
    # Try alternative parsers if default fails
    try:
        data = pd.read_csv(file_path, engine="python")  # Use Python's built-in parser
    except pd.errors.ParserError:
        try:
            data = pd.read_csv(file_path, sep=";", engine="python")  # Try semicolon delimiter
        except pd.errors.ParserError:
            print("Error loading data. Check file format and delimiters.")
            raise

# Explore data (if successfully loaded)
data.head()
data.info()
data.describe()
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])
print("Data types:")
print(data.dtypes)
print("Unique values:")
print(data.nunique())

# Check for missing values
data.isnull().sum()

