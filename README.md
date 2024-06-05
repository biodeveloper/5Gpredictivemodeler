

# 5Gpredictivemodeler
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>5G Predictive Modeling - Code Overview</title>
</head>
<body>
  <h1>5G Predictive Modeling - Code Overview</h1>

  <h2>1. Brief Code Description</h2>

  <p>This code utilizes Python version 3.12.1 (64-bit) and consists of two Python scripts: `main.py` and `Data_Visualization.py`.</p>

  <p>**main.py** performs the following tasks:</p>

  <ul>
    <li>Reads data from a CSV file (`QualityOfService5GDataset-3.csv`) in the `data` folder.</li>
    <li>Creates a MySQL database to store the results of XGBoost and Random Forest executions.</li>
    <li>Saves generated graphs (statistical analysis, SHAP Feature importance bar chart, and SHAP summary plot) to the `graphs` folder.</li>
  </ul>

  <p>**Data_Visualization.py** performs the following tasks:</p>

  <ul>
    <li>Reads the MySQL database and creates 3 new graphs to compare two machine learning models: XGBoost and Random Forest.</li>
  </ul>

  <h2>2. Libraries</h2>

  <p>The code uses the following core libraries:</p>

  <ul>
    <li>**os:** For file and folder manipulation.</li>
    <li>**pandas:** For data processing using DataFrames.</li>
    <li>**numpy:** For numerical operations with arrays.</li>
    <li>**matplotlib.pyplot:** For creating graphs.</li>
    <li>**mysql.connector:** For connecting to and managing a MySQL database.</li>
    <li>**shap:** For SHAP (SHapley Additive exPlanations) analysis and feature importance visualization.</li>
    <li>**json:** For converting data to JSON format.</li>
  </ul>

  <h2>3. Functionality</h2>

  <h3>3.1 Database Connection</h3>

  <p>The code connects to the MySQL database "5Gpredictivemodeler".</p>

  <h3>3.2 Data Retrieval</h3>

  <p>It retrieves data from the `model_results` table for both XGBoost and Random Forest models, including graph data.</p>

  <h3>3.3 Graph Creation</h3>

  <ul>
    <li>**SHAP Bar Plot:** Creates a combined SHAP bar plot, showing feature importance for both models.</li>
    <li>**SHAP Summary Plot:** Creates a combined SHAP summary plot, showing the distribution of SHAP values for each feature, grouped by model.</li>
    <li>**Model Performance Comparison:** Creates a graph comparing performance metrics (RMSE, MSE, R^2) for both models.</li>
  </ul>

  <h3>3.4 Graph Saving</h3>

  <p>Saves the graphs to the "graphs" folder, with file names that include a timestamp.</p>

  <h2>4. Purpose</h2>

  <p>This code facilitates the visualization and comparison of two machine learning models trained on the same dataset, leveraging stored data and SHAP graphs.</p>
</body>
</html>


