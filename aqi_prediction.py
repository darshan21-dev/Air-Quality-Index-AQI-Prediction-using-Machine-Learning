# aqi_prediction.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# import os # Not needed currently for file path handling
# import requests # Not needed currently for API usage
# import json # Not needed currently for API usage

print("AQI Prediction Project Started...")

# --------------------------------------------------------------------------------
# Step 2: Loading real AQI data (for training)
# --------------------------------------------------------------------------------
print("\nStep 2: Loading historical AQI data (for training)...")

# Datafile name and path
# Ensure that city_day.csv file is in your project folder or in './archive/'
DATA_FILE = './archive/city_day.csv'

try:
    data = pd.read_csv(DATA_FILE)
    print(f"Data successfully loaded from '{DATA_FILE}'.")
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' file not found. Please ensure the file is in the project folder.")
    print("You need to download the 'Air Quality in India' dataset from Kaggle and place the 'city_day.csv' file in the project folder or its 'archive' sub-folder.")
    exit() # Exit program if file not found

print("\nFirst 5 data entries:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Handling Missing Values in real data
# This dataset may have NaN (Not a Number) values for many pollutants.
# For simplicity, we will fill them with the mean.
# More complex methods are used in real projects.
print("\nChecking for Missing Values:")
print(data.isnull().sum())

# Drop unnecessary columns
# We are dropping City, Date, AQI_Bucket here to use the remaining columns as features and target.
data = data.drop(columns=['AQI_Bucket', 'City', 'Date'], errors='ignore')

# Fill missing values with the mean of the features
# This loop will fill NaN values in all numeric columns.
for col in data.columns:
    if data[col].dtype in ['float64', 'int64'] and data[col].isnull().any():
        data[col] = data[col].fillna(data[col].mean())

print("\nAfter filling missing values (NaNs filled):")
print(data.isnull().sum()) # There should be no NaNs in numeric columns now

# Ensure AQI is within the range of 0 to 500
data['AQI'] = np.clip(data['AQI'], 0, 500)

# --------------------------------------------------------------------------------
# Step 3: Data Preprocessing
# --------------------------------------------------------------------------------
print("\nStep 3: Data Preprocessing...")

# Separate features (X) and target (y)
X = data.drop('AQI', axis=1) # All columns except AQI are features
y = data['AQI']

# Store X_train column names. These features will be used for prediction.
X_feature_names_for_prediction = X.columns.tolist()

# Split data into training and testing sets
# 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Data Scaling (standardize features)
# Use StandardScaler to scale data so all features are in a similar range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 scaled training data (X_train_scaled) entries:")
print(X_train_scaled[:5])

# --------------------------------------------------------------------------------
# Step 4: Model Selection (RandomForestRegressor)
# --------------------------------------------------------------------------------
print("\nStep 4: Model Selection (RandomForestRegressor)...")

# Select RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 will use all cores
print("Selected Model: RandomForestRegressor")

# --------------------------------------------------------------------------------
# Step 5: Model Training
# --------------------------------------------------------------------------------
print("\nStep 5: Model is being trained...")
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --------------------------------------------------------------------------------
# Step 6: Model Evaluation
# --------------------------------------------------------------------------------
print("\nStep 6: Model Evaluation...")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Visualization of Actual vs Predicted AQI values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Ideal prediction line
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.grid(True)
plt.show()

# Visualization of Feature Importance
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance - AQI Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# --------------------------------------------------------------------------------
# Step 7: Prediction for new data (with manual data)
# --------------------------------------------------------------------------------
print("\nStep 7: Prediction for new data...")

# Create a DataFrame for new data.
# IMPORTANT: This DataFrame must only contain the same columns as X_train, and in the same order.
# The city_day.csv does not have Temperature, Humidity, Wind_Speed columns, so they are removed here.
# And city_day.csv has NO, NOx, NH3, Benzene, Toluene, Xylene columns, so they are included here.

# Here, we are using mean values from the training data as an example for new data.
# In a real application, you would obtain these values from somewhere (e.g., sensors, manual input).
example_new_data = {
    'PM2.5': [45.0, 120.0, 25.0],
    'PM10': [70.0, 180.0, 40.0],
    'NO': [X['NO'].mean(), 50.0, 10.0], # Example using mean and some custom values
    'NO2': [30.0, 90.0, 18.0],
    'NOx': [X['NOx'].mean(), 80.0, 25.0],
    'NH3': [X['NH3'].mean(), 15.0, 5.0],
    'CO': [2.5, 7.0, 1.2],
    'SO2': [15.0, 60.0, 8.0],
    'O3': [50.0, 150.0, 35.0],
    'Benzene': [X['Benzene'].mean(), 10.0, 1.5],
    'Toluene': [X['Toluene'].mean(), 20.0, 3.0],
    'Xylene': [X['Xylene'].mean(), 5.0, 0.5]
}

# Ensure that the order of columns matches the order of columns in X_train.
new_air_data = pd.DataFrame(example_new_data, columns=X_feature_names_for_prediction)

print("\nNew Data:")
print(new_air_data)

# New data must also be scaled like the training data
# This step will now not give an error because feature names match.
new_air_data_scaled = scaler.transform(new_air_data)

# Make predictions on the new data
new_aqi_predictions = model.predict(new_air_data_scaled)

print("\nPredicted Air Quality Index (AQI) for new data:")
for i, pred_aqi in enumerate(new_aqi_predictions):
    print(f"Sample {i+1}: AQI = {pred_aqi:.2f}")

# Optional: To show which AQI category it falls into
def get_aqi_category(aqi_value):
    if 0 <= aqi_value <= 50:
        return "Good"
    elif 51 <= aqi_value <= 100:
        return "Satisfactory"
    elif 101 <= aqi_value <= 200:
        return "Moderately Polluted"
    elif 201 <= aqi_value <= 300:
        return "Poor"
    elif 301 <= aqi_value <= 400:
        return "Very Poor"
    elif 401 <= aqi_value <= 500:
        return "Severe"
    else:
        return "Beyond Index"

print("\nPredicted AQI Categories:")
for i, pred_aqi in enumerate(new_aqi_predictions):
    category = get_aqi_category(pred_aqi)
    print(f"Sample {i+1}: AQI = {pred_aqi:.2f} ({category})")

print("\nProject Complete!")