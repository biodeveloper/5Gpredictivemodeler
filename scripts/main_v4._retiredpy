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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
# visualization and interpretability
import matplotlib.pyplot as plt
import shap
import datetime  # Import datetime for timestamp
# database connection
import mysql.connector
import json

# Get current working directory
cwd = os.getcwd()
print("Current working directory path:", cwd)

# Construct file paths
data_path = os.path.join(cwd, "data", "QualityOfService5GDataset-3.csv")  # Path to data file
graph_path = os.path.join(cwd, "graphs")  # Path to graphs directory
db_path = os.path.join(cwd, "data", "5Gpredictivemodeler.db")  # Path to database file

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

# Get current timestamp for filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Pie chart for application type distribution 
app_counts = data["application_type"].value_counts() 
plt.figure(figsize=(8, 6)) 
plt.pie(app_counts.values, labels=app_counts.index, autopct=lambda p: f"{p*sum(app_counts.values)/100:.0f} ({p:.0f}%)") 
plt.title("Distribution of Application Types")
plt.savefig(os.path.join(graph_path, f"application_types_piechartplot_{timestamp}.png"), dpi=300)

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
    plt.savefig(os.path.join(graph_path, f"{feature}_histogram_{timestamp}.png"), dpi=300)

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

# --- Database Connection and Setup ---
try:
    # Connect without specifying the database initially
    mydb = mysql.connector.connect(
        host="localhost",
        user="5gpred",  # Replace with your MySQL username
        password="wyw5445hkwjb@y7^"  # Replace with your MySQL password
    )

    mycursor = mydb.cursor()

    # Check if the database exists
    mycursor.execute("SHOW DATABASES LIKE '5Gpredictivemodeler'")
    database_exists = mycursor.fetchone()

    # Create the database only if it doesn't exist
    if not database_exists:
        mycursor.execute("CREATE DATABASE 5Gpredictivemodeler")
        print("Database '5Gpredictivemodeler' created.")

    # Now select the database
    mydb.database = "5Gpredictivemodeler"  

    # Create table for model results
    mycursor.execute("""
    CREATE TABLE IF NOT EXISTS model_results (
        model_name VARCHAR(255) PRIMARY KEY,
        timestamp DATETIME,
        best_parameters TEXT,
        cross_val_rmse FLOAT,
        test_mse FLOAT,
        test_r2 FLOAT,
        test_rmse FLOAT,
        all_parameters TEXT
    )
    """)

except mysql.connector.Error as err:
    print(f"Database error: {err}")
    exit(1)

# --- Model Selection from Terminal ---
while True:
    model_choice = input("Choose a regression model (1-XGBoost, 2-Random Forest, 3-Neural Network): ")
    if model_choice in ['1', '2', '3']:
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

models = {
    "1": ("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42), {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 500, 1000],
        'colsample_bytree': [0.5, 0.8, 1.0]
    }),
    "2": ("Random Forest", RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 500, 1000],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }),
    "3": ("Neural Network", MLPRegressor(random_state=42, max_iter=500), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    })
}

# --- Model Training and Evaluation ---
model_name, model, param_grid = models[model_choice]
print(f"--- Training {model_name} ---")

# Hyperparameter Tuning with GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)

# Use root_mean_squared_error function for RMSE
rmse = root_mean_squared_error(y_test, y_pred) 

mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R^2: {r2:.2f}\n")
# Get the best score from grid_search 
best_score = grid_search.best_score_

# Evaluate on test set (Calculate MSE here)
y_pred = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)  # Calculate RMSE on the test set
mse = mean_squared_error(y_test, y_pred)  # Calculate MSE on the test set
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# --- Store Results in Database ---
try:
    all_params = best_model.get_params()  # Get all parameters of the best model
    sql = """
    REPLACE INTO model_results (model_name, timestamp, best_parameters, cross_val_rmse, 
                               test_mse, test_r2, test_rmse, all_parameters) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    val = (model_name, datetime.datetime.now(), str(grid_search.best_params_), (-best_score)**0.5, 
           mse, r2, rmse, str(all_params))

    mycursor.execute(sql, val)
    mydb.commit()

    print("Model results saved to database.")

except mysql.connector.Error as err:
    print(f"Database error: {err}")


# --- SHAP Analysis ---
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# Create legend text for best parameters and metrics
legend_text = (f"Best Hyperparameters: {grid_search.best_params_}\n"
               f"Best Cross-Validation Score (RMSE): {(-best_score)**0.5:.2f}\n" # RMSE from cross-validation
               f"Training MSE: {mse:.2f}\n"
               f"Training R^2: {r2:.2f}\n"
               f"RMSE: {rmse:.2f}")  # RMSE on the test set

# --- Visualize SHAP Bar Plot with legend ---
plt.figure(figsize=(12, 6))
shap.plots.bar(shap_values, show=False)
plt.title(f"SHAP Feature Importance - {model_name}")

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

# Add legend below the plot
plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=10, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
plt.savefig(os.path.join(graph_path, f"shap_bar_plot_{model_name}_{timestamp}.png"), dpi=300)
plt.show()

# --- Visualize SHAP Summary Plot with legend ---
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title(f"SHAP Summary Plot - {model_name}")

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

# Add legend below the plot
plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=10, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
plt.savefig(os.path.join(graph_path, f"shap_summary_plot_{model_name}_{timestamp}.png"), dpi=300)
plt.show()