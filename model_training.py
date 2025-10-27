import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# ---- Set dataset path ----
dataset_path = r"C:\Users\Sri Varsha\Desktop\edunet"
file_name = "copy of water_data 1.csv"
file_path = os.path.join(dataset_path, file_name)

# ---- Load dataset ----
print("✅ Reading:", file_path)
data = pd.read_csv(file_path)

# ---- Select relevant columns ----
features = ['pH', 'moisture', 'nitrogen', 'phosphorus', 'potassium', 'temperature']
existing_features = [col for col in features if col in data.columns]
print("✅ Using features:", existing_features)

# ---- Drop rows with missing values in selected columns ----
data = data[existing_features].dropna()

# ---- Create synthetic target variable (Groundwater Potential %) ----
weights = {
    'moisture': 0.4,
    'pH': 0.2,
    'nitrogen': 0.15,
    'phosphorus': 0.1,
    'potassium': 0.1,
    'temperature': 0.05
}
data['groundwater_potential'] = sum(data[col] * weights[col] for col in existing_features if col in weights)

# ---- Split data ----
X = data[existing_features]
y = data['groundwater_potential']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train model ----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ---- Evaluate model ----
y_pred = model.predict(X_test_scaled)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ---- Save model and scaler ----
joblib.dump(model, os.path.join(dataset_path, 'hydro_model.pkl'))
joblib.dump(scaler, os.path.join(dataset_path, 'scaler.pkl'))
print("✅ Model and Scaler saved successfully!")

# ---- Predict function for new inputs ----
def predict_groundwater(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[[col for col in existing_features if col in input_dict]]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Example usage:
example_input = {
    'pH': 6.5,
    'moisture': 30,
    'temperature': 25
}
predicted = predict_groundwater(example_input)
print(f"Predicted Groundwater Potential: {predicted:.2f}%")
