# Data Loading and Exploration
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde  # Import gaussian_kde for density estimation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
# scikit-learn models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# visualization and interpretability
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
X = pd.get_dummies(X, columns=['application_type'])
X = X.fillna(0)
scaler = MinMaxScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
models = {
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Neural Network": MLPRegressor(random_state=42, max_iter=500)  # Increase max_iter for NN
}

param_grids = {
    "XGBoost": {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.5, 0.8, 1.0]
    },
    "Random Forest": {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    "Neural Network": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }
}

results = {}

for model_name, model in models.items():
    print(f"--- Training {model_name} ---")
    
    # Hyperparameter Tuning with GridSearchCV
    grid_search = GridSearchCV(model, param_grids[model_name], cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {"RMSE": rmse, "MAPE": mape, "R^2": r2, "Best Model": best_model}
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    print(f"R^2: {r2:.2f}\n")

    # --- SHAP Analysis ---
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X_test)

    # Visualize feature importance with legend
    plt.figure(figsize=(12, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, f"shap_bar_plot_{model_name}.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path, f"shap_summary_plot_{model_name}.png"), dpi=300)
    plt.show() 


# --- Model Comparison ---
print("--- Model Comparison ---")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}")
    print(f"  R^2: {metrics['R^2']:.2f}")