#XGBoost Regressor Model Training
# Create XGBoost regressor model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)