# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap

# Load the dataset
df = pd.read_excel('Quality of Service 5G.xlsx')

# Drop the user ID column
df.drop('user ID', axis=1, inplace=True)

# Convert the 'application type' column to dummy variables
df = pd.get_dummies(df, columns=['application type'])

# Scale all numerical columns (except the target)
scaler = MinMaxScaler()
df[['signal strength (dBm)', 'latency (msec)', 'required bandwidth (Mbps)', 'allocated bandwidth (Mbps)']] = scaler.fit_transform(df[['signal strength (dBm)', 'latency (msec)', 'required bandwidth (Mbps)', 'allocated bandwidth (Mbps)']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('latency (msec)', axis=1), df['latency (msec)'], test_size=0.2, random_state=42)

# Train the models
models = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), MLPRegressor()]
for model in models:
   model.fit(X_train, y_train)

# Evaluate the models
for model in models:
   print(f'Model: {model.__class__.__name__}')
   print(f'RMSE: {mean_squared_error(y_test, model.predict(X_test), squared=False)}')
   print(f'MAPE: {mean_absolute_percentage_error(y_test, model.predict(X_test))}\n')

# Calculate feature importances for the Random Forest and Gradient Boosting models
importances = {}
for model in [RandomForestRegressor(), GradientBoostingRegressor()]:
   model.fit(X_train, y_train)
   importances[model.__class__.__name__] = model.feature_importances_

# Plot the feature importances
for model, importances in importances.items():
   plt.figure(figsize=(10, 5))
   plt.bar(df.drop('latency (msec)', axis=1).columns, importances)
   plt.title(f'Feature Importances for {model}')
   plt.show()

# Calculate SHAP values for the Random Forest and Gradient Boosting models
explainer = shap.Explainer(RandomForestRegressor())
shap_values_rf = explainer.shap_values(X_train)
explainer = shap.Explainer(GradientBoostingRegressor())
shap_values_gb = explainer.shap_values(X_train)

# Plot the SHAP values
for shap_values, model in zip([shap_values_rf, shap_values_gb], ['Random Forest', 'Gradient Boosting']):
   plt.figure(figsize=(10, 5))
   shap.summary_plot(shap_values, X_train)
   plt.title(f'SHAP Values for {model}')
   plt.show()