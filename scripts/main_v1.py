# Data Loading and Exploration
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde  # Import gaussian_kde for density estimation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
# scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

# Get current working directory
cwd = os.getcwd()
print("Current working directory path:", cwd)

# Construct file paths
data_path = os.path.join(cwd, "data", "QualityOfService5GDataset-3.csv")  # Path to data file
graph_path = os.path.join(cwd, "graphs") # Path to graphs directory

# Check if data file exists
if os.path.isfile(data_path):
    print("Data file exists.")
else:
    print("Data file does not exist.")

# Load data from CSV file with error handling and type specification
try:
    data = pd.read_csv(data_path, dtype={'User_ID': 'string', 
                                          'application_type': 'string', 
                                          'signal_strength(dBm)': 'float64', 
                                          'latency(msec)': 'int64', 
                                          'required_bandwidth(Mbps)': 'int64', 
                                          'allocated_bandwidth(Mbps)': 'int64', 
                                          'resource_allocation': 'int64'})
except pd.errors.ParserError:
    print("Trying alternative parsers...")
    try:
        data = pd.read_csv(data_path, engine="python", dtype={'User_ID': 'string',
                                                             'application_type': 'string',
                                                             'signal_strength(dBm)': 'float64',
                                                             'latency(msec)': 'int64',
                                                             'required_bandwidth(Mbps)': 'int64',
                                                             'allocated_bandwidth(Mbps)': 'int64',
                                                             'resource_allocation': 'int64'}) 
    except pd.errors.ParserError:
        print("Error loading data. Check file format and delimiters.")
        raise

# Explore Data
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

# Data Visualization
# Create graphs directory if it doesn't exist
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

# Pie chart for application type distribution 
app_counts = data["application_type"].value_counts() 
plt.figure(figsize=(8, 6)) 
plt.pie(app_counts.values, labels=app_counts.index, autopct=lambda p: f"{p*sum(app_counts.values)/100:.0f} ({p:.0f}%)") 
plt.title("Distribution of Application Types")
plt.savefig(os.path.join(graph_path, "application_types_piechartplot.png"), dpi=300)

# Histograms for numerical features
numerical_features = ["signal_strength(dBm)", "latency(msec)", "required_bandwidth(Mbps)", "allocated_bandwidth(Mbps)", "resource_allocation"]

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    data[feature].hist(bins=10, density=True, edgecolor='black')

    # Density curve using KDE
    density = gaussian_kde(data[feature])
    x = np.linspace(data[feature].min(), data[feature].max(), 200) 
    plt.plot(x, density(x), color='red', linewidth=2, label="Εκτιμώμενη Κατανομή")
    
    plt.legend()
    plt.title(f"Κατανομή της Μεταβλητής '{feature}'", fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Πυκνότητα", fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(graph_path, f"{feature}_histogram.png"), dpi=300)


# Data Preprocessing
y = data['latency(msec)'] 

X = data.drop(['User_ID', 'latency(msec)'], axis=1)

# One-hot encode 'application type'
X = pd.get_dummies(X, columns=['application_type'])

# Handle missing values (fill with 0)
X = X.fillna(0)

# Min-Max scaling for numerical features
scaler = MinMaxScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# XGBoost Regressor with Hyperparameter Tuning and SHAP
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [0.5, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate model performance on the training set
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Create legend text for best parameters and metrics
legend_text = (f"Best Hyperparameters: {best_params}\n"
               f"Best Cross-Validation Score (RMSE): {(-best_score)**0.5:.2f}\n"
               f"Training MSE: {mse:.2f}\n"
               f"Training R^2: {r2:.2f}")

# SHAP Analysis for Feature Importance
explainer = shap.Explainer(best_model)
shap_values = explainer(X)

# Visualize SHAP Bar Plot with legend
plt.figure(figsize=(12, 6))
shap.plots.bar(shap_values, show=False)
plt.legend([legend_text], loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "shap_bar_plot_with_legend.png"), dpi=300)
plt.show()

# Visualize SHAP Summary Plot with legend
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X, show=False)
plt.legend([legend_text], loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "shap_summary_plot_with_legend.png"), dpi=300)
plt.show()