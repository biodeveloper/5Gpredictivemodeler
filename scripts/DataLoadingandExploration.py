
#Data Loading and Exploration
import os
import pandas as pd
import numpy as np
from scipy.stats import norm  # Import norm from scipy.stats for density curve
from scipy.stats import gaussian_kde  # Import gaussian_kde for density estimation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#scikit-learn
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Get current working directory
cwd = os.getcwd()


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
                              'latency(ms)': 'int64',  # Specify float for latency
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
                              'latency(ms)': 'int64',  # Specify float for latency
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
                              'latency(ms)': 'int64',  # Specify float for latency
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
file_path2 = os.path.join(cwd, "graphs")
if not os.path.exists(file_path2):
    os.makedirs(file_path2)


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
save_path = os.path.join(file_path2, f"application_types_piechartplot.png")
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
    save_path = os.path.join(file_path2, f"{feature}_histogram.png")
    plt.savefig(save_path, dpi=300)

#    plt.show() 

# Data Preprocessing
# One-hot encode categorical feature "application type"
X = pd.get_dummies(data, columns=["application_type"])

# Fill missing values with 0 (if any)
X = X.fillna(0)

# Min-Max scaling for features (excluding target)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on training data

# Create XGBoost regressor model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)  # Objective remains the same

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)