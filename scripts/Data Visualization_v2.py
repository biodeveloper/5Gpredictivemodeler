import os
import json
import datetime
import matplotlib.pyplot as plt
import mysql.connector
import shap
import numpy as np

# Get current working directory
cwd = os.getcwd()
print("Current working directory path:", cwd)

# Construct file paths
graph_path = os.path.join(cwd, "graphs")  # Path to graphs directory

# Database Connection
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="5gpred",  # Replace with your MySQL username
        password="wyw5445hkwjb@y7^",  # Replace with your MySQL password
        database="5Gpredictivemodeler"
    )

    mycursor = mydb.cursor()

except mysql.connector.Error as err:
    print(f"Database error: {err}")
    exit(1)

# --- Retrieve data for XGBoost and Random Forest ---
model_names = ["XGBoost", "Random Forest"]
results = {}
for model_name in model_names:
    mycursor.execute("SELECT * FROM model_results WHERE model_name = %s", (model_name,))
    result = mycursor.fetchone()
    if result:
        results[model_name] = {
            "timestamp": result[1],
            "best_parameters": json.loads(result[2]),
            "cross_val_rmse": result[3],
            "test_mse": result[4],
            "test_r2": result[5],
            "test_rmse": result[6],
            "all_parameters": json.loads(result[7]),
            "plot_data": json.loads(result[8]),
            "X_test_columns": json.loads(result[9])
        }
        
# --- 1. Combined SHAP Bar Plot ---
plt.figure(figsize=(12, 6))

bar_width = 0.35
feature_indices = np.arange(len(results["XGBoost"]["X_test_columns"]))  # Assuming same features for both models
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Define timestamp here
for i, model_name in enumerate(model_names):
    shap_values = np.array(results[model_name]["plot_data"]["shap_bar"])
    base_values = np.array(results[model_name]["plot_data"].get("shap_base_values", 0))
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=results[model_name]["plot_data"]["shap_summary"],
        feature_names=results[model_name]["X_test_columns"]
    )

    # Plot bars side by side
    plt.barh(feature_indices + i * bar_width, shap_explanation.values.mean(0), 
             bar_width, label=model_name, color=['blue', 'red'][i])  # Assign colors based on model

plt.xlabel("Mean(|SHAP value|)")  # Update x-axis label
plt.ylabel("Features")  # Update y-axis label
plt.title("SHAP Feature Importance - XGBoost vs. Random Forest")
plt.yticks(feature_indices + bar_width / 2, results["XGBoost"]["X_test_columns"])  # Set y-tick labels
plt.legend()

# Adjust layout for legend
plt.subplots_adjust(left=0.3)  # Adjust left margin to accommodate feature names

# Add legend with model details
legend_text = [f"Best Hyperparameters: {results[model_name]['best_parameters']}\n"
               f"Best Cross-Validation Score (RMSE): {results[model_name]['cross_val_rmse']:.2f}\n"
               f"Training MSE: {results[model_name]['test_mse']:.2f}\n"
               f"Training R^2: {results[model_name]['test_r2']:.2f}\n"
               f"RMSE: {results[model_name]['test_rmse']:.2f}"
               for model_name in model_names]
plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=10, 
            bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

plt.savefig(os.path.join(graph_path, f"combined_shap_bar_plot_{timestamp}.png"), dpi=300)
plt.show()


# --- 3. Model Performance Comparison Plot ---
plt.figure(figsize=(10, 6))
metrics = ["cross_val_rmse", "test_mse", "test_r2", "test_rmse"]
bar_width = 0.35
index = np.arange(len(metrics))

for i, model_name in enumerate(model_names):
    plt.bar(index + i * bar_width, [results[model_name][metric] for metric in metrics], 
            bar_width, label=model_name)

plt.xlabel("Metrics")
plt.ylabel("Values")
plt.title("Model Performance Comparison - XGBoost vs. Random Forest")
plt.xticks(index + bar_width / 2, metrics)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"model_performance_comparison_{timestamp}.png"), dpi=300)
plt.show()

