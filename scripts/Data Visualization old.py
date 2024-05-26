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
            "plot_data": json.loads(result[8])
        }

# --- 1. Combined SHAP Bar Plot ---
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(model_names):
    shap_values = np.array(results[model_name]["plot_data"]["shap_bar"])
    shap.plots.bar(shap.Explanation(values=shap_values, 
                                   base_values=0, 
                                   data=results[model_name]["plot_data"]["shap_summary"], 
                                   feature_names=X_test.columns), 
                   show=False)
    plt.title(f"SHAP Feature Importance - {model_name} vs. Random Forest")

# Adjust layout for legend
plt.subplots_adjust(bottom=0.3) 

# Add legend with model details
legend_text = [f"{model_name}:\n"
               f"  Best Parameters: {results[model_name]['best_parameters']}\n"
               f"  Cross-Val RMSE: {results[model_name]['cross_val_rmse']:.2f}\n"
               f"  Test MSE: {results[model_name]['test_mse']:.2f}\n"
               f"  Test R^2: {results[model_name]['test_r2']:.2f}\n"
               f"  Test RMSE: {results[model_name]['test_rmse']:.2f}"
               for model_name in model_names]
plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=10, 
            bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(graph_path, f"combined_shap_bar_plot_{timestamp}.png"), dpi=300)
plt.show()

# --- 2. Combined SHAP Summary Plot ---
plt.figure(figsize=(12, 6))
for i, model_name in enumerate(model_names):
    shap_values = np.array(results[model_name]["plot_data"]["shap_bar"])
    shap.summary_plot(shap.Explanation(values=shap_values, 
                                   base_values=0, 
                                   data=results[model_name]["plot_data"]["shap_summary"], 
                                   feature_names=X_test.columns), 
                       show=False, 
                       alpha=0.5 + i * 0.5)  # Adjust transparency for overlap

plt.title("SHAP Summary Plot - XGBoost vs. Random Forest")

# Adjust layout for legend
plt.subplots_adjust(bottom=0.3) 

plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=10, 
            bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})

plt.savefig(os.path.join(graph_path, f"combined_shap_summary_plot_{timestamp}.png"), dpi=300)
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