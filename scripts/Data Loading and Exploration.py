
#Data Loading and Exploration

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv("Quality Of Service 5G Dataset.csv")

# Explore data
data.head()
data.info()
data.describe()

# Check for missing values
data.isnull().sum()