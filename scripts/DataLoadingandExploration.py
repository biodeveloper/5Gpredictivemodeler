

#Data Loading and Exploration
import os
import pandas as pd
import numpy as np
from scipy.stats import norm  # Import norm from scipy.stats for density curve
from scipy.stats import gaussian_kde  # Import gaussian_kde for density estimation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
#scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

# Get current working directory
cwd = os.getcwd()
print("current working directory path:", cwd)

# Construct file path
file_path = os.path.join(cwd, "data\QualityOfService5GDataset-3.csv")
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
    data = pd.read_csv(file_path, dtype={'User_ID': 'string',  # Specify integer for User_ID
                              'application_type': 'string',  # Specify category for application type
                              'signal_strength(dBm)': 'float64',  # Specify float for signal strength
                              'latency(msec)': 'int64',  # Specify float for latency
                              'required_bandwidth(Mbps)': 'int64',  # Specify float for bandwidth
                              'allocated_bandwidth(Mbps)': 'int64',  # Specify float for allocated bandwidth
                              'resource_allocation': 'int64'})
except pd.errors.ParserError:
    # Try alternative parsers if default fails
    print("try 2nd parser")
    try:
        data = pd.read_csv(file_path, engine="python", dtype={'User_ID': 'string',  # Specify integer for User_ID
                              'application_type': 'string',  # Specify category for application type
                              'signal_strength(dBm)': 'float64',  # Specify float for signal strength
                              'latency(msec)': 'int64',  # Specify float for latency
                              'required_bandwidth(Mbps)': 'int64',  # Specify float for bandwidth
                              'allocated_bandwidth(Mbps)': 'int64',  # Specify float for allocated bandwidth
                              'resource_allocation': 'int64'})  # Use Python's built-in parser
    except pd.errors.ParserError:
        # Try alternative parsers if default fails
        print("try 3rd parser")
        try:
            data = pd.read_csv(file_path, sep=";", engine="python", dtype={'User_ID': 'string',  # Specify integer for User_ID
                              'application_type': 'string',  # Specify category for application type
                              'signal_strength(dBm)': 'float64',  # Specify float for signal strength
                              'latency(msec)': 'int64',  # Specify float for latency
                              'required_bandwidth(Mbps)': 'int64',  # Specify float for bandwidth
                              'allocated_bandwidth(Mbps)': 'int64',  # Specify float for allocated bandwidth
                              'resource_allocation': 'int64'})  # Try semicolon delimiter
        except pd.errors.ParserError:
            # Try alternative parsers if default fails
            print("try 4th parser")
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

#Data Visualization
# Create directory for graphs if it doesn't exist
graph_path = os.path.join(cwd, "graphs")
if not os.path.exists(graph_path):
    os.makedirs(graph_path)


# Pie chart for application type distribution with counts and percentages
app_counts = data["application_type"].value_counts() # Μετράμε την συχνότητα εμφάνισης κάθε τύπου εφαρμογής
plt.figure(figsize=(8, 6)) # Ορίζουμε το μέγεθος του γραφήματος
plt.pie(app_counts.values, labels=app_counts.index, autopct=lambda p: f"{p*sum(app_counts.values)/100:.0f} ({p:.0f}%)") 
# Δημιουργούμε το γράφημα πίτας
# app_counts.values: Παρέχουμε τις τιμές (μεγέθη των τμημάτων)
# app_counts.index: Παρέχουμε τις ετικέτες (ονόματα τύπων εφαρμογών)
# autopct: Ορίζουμε μια συνάρτηση για τη μορφοποίηση των ετικετών ποσοστού και αριθμού
plt.title("Distribution of Application Types") # Τίτλος του γραφήματος
# Save plot to PNG file
save_path = os.path.join(graph_path, f"application_types_piechartplot.png")
plt.savefig(save_path, dpi=300)  # Save with 300 DPI for higher resolution
#plt.show() # Εμφανίζουμε το γράφημα

# Histograms for numerical features
# Histograms with density lines and legend
# Separate histograms with density lines and legend
numerical_features = ["signal_strength(dBm)", "latency(msec)", "required_bandwidth(Mbps)", "allocated_bandwidth(Mbps)", "resource_allocation"]

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    data[feature].hist(bins=10, density=True, edgecolor='black')

    # Calculate and plot density curve using KDE
    density = gaussian_kde(data[feature])
    x = np.linspace(data[feature].min(), data[feature].max(), 200)  # More points for smoother curve
    plt.plot(x, density(x), color='red', linewidth=2, label="Εκτιμώμενη Κατανομή")
    
    plt.legend()
    plt.title(f"Κατανομή της Μεταβλητής '{feature}'", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Πυκνότητα", fontsize=12)
    plt.grid(True)

    # Save plot to PNG file
    #if file exists, overwrite it
    save_path = os.path.join(graph_path, f"{feature}_histogram.png")
    plt.savefig(save_path, dpi=300)

#    plt.show() 

# Data Preprocessing
# One-hot encode categorical feature "application type"
# Separate target variable
y = data['latency(msec)'] 

# Drop unnecessary columns (user ID and target variable)
X = data.drop(['User_ID', 'latency(msec)'], axis=1)

# One-hot encode 'application type'
X = pd.get_dummies(X, columns=['application_type'])

# Handle missing values (fill with 0)
X = X.fillna(0)

# Min-Max scaling for numerical features
scaler = MinMaxScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Now we have X (preprocessed features) and y (target variable) ready for model training
# XGBoost Regressor with Hyperparameter Tuning and SHAP

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

# Create XGBoost regressor model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Perform GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best model and its performance
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Cross-Validation Score (RMSE):", (-best_score)**0.5)  # Convert from negative MSE

# Evaluate model performance on the training set
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Training MSE:", mse)
print("Training R^2:", r2)

# SHAP Analysis for Feature Importance
explainer = shap.Explainer(best_model)
shap_values = explainer(X)

# Visualize feature importance
shap.plots.bar(shap_values)
shap.summary_plot(shap_values, X)