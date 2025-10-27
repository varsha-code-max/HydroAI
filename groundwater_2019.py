import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# ---- 1️⃣ Load dataset files ----
dataset_path = r"C:\Users\Sri Varsha\Desktop\edunet"

# File names (make sure .csv is present in the file name)
file1 = os.path.join(dataset_path, "ground_water_quality_2018_post.csv")
file2 = os.path.join(dataset_path, "groundwater_recharge_feasibility.csv")

# ---- 2️⃣ Read the datasets ----
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("✅ Datasets loaded successfully!")
print("Groundwater Quality columns:", df1.columns.tolist())
print("Recharge Feasibility columns:", df2.columns.tolist())

# ---- 3️⃣ Merge both datasets ----
# If both have 'district', 'mandal', or 'village' columns, merge on them
common_cols = [col for col in ['district', 'mandal', 'village'] if col in df1.columns and col in df2.columns]
if common_cols:
    df = pd.merge(df1, df2, on=common_cols, how='inner')
else:
    df = pd.concat([df1, df2], axis=0, ignore_index=True)

print(f"\n✅ Combined dataset shape: {df.shape}")

# ---- 4️⃣ Select relevant features ----
# Include common soil/water quality and environmental parameters
feature_candidates = [
    'pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3 ', 'SO4', 
    'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC  meq  / L',
    'soil_moisture', 'temperature', 'rainfall'
]

available_features = [f for f in feature_candidates if f in df.columns]
print("\n✅ Using available features:", available_features)

# ---- Target variable ----
target = 'gwl' if 'gwl' in df.columns else None
if target is None:
    raise ValueError("❌ 'gwl' (groundwater level) column not found in dataset. Please check the column name.")

# ---- 5️⃣ Handle missing data ----
df = df.dropna(subset=[target])
X = df[available_features].fillna(df.mean(numeric_only=True))
y = df[target]

# ---- 6️⃣ Split dataset ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 7️⃣ Scale features ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- 8️⃣ Train model ----
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# ---- 9️⃣ Evaluate model ----
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# ---- 🔟 Prediction function ----
def predict_groundwater(input_dict):
    """
    Example input:
    input_dict = {
        'pH': 7.2, 'E.C': 0.56, 'TDS': 320, 'CO3': 14, 'HCO3': 120,
        'Cl': 50, 'F': 1.0, 'NO3 ': 22, 'SO4': 18, 'Na': 42,
        'K': 4.2, 'Ca': 36, 'Mg': 14, 'T.H': 200, 'SAR': 2.1,
        'RSC  meq  / L': 0.9, 'soil_moisture': 25, 'temperature': 30, 'rainfall': 150
    }
    """
    input_df = pd.DataFrame([input_dict])
    # Keep only columns available in the training data
    input_df = input_df[[col for col in available_features]]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# ---- 🔬 Example Prediction ----
example_input = {
    'pH': 7.7, 'E.C': 0.52, 'TDS': 310, 'CO3': 15, 'HCO3': 145,
    'Cl': 60, 'F': 1.1, 'NO3 ': 25, 'SO4': 20, 'Na': 40,
    'K': 4, 'Ca': 38, 'Mg': 12, 'T.H': 210, 'SAR': 2.3,
    'RSC  meq  / L': 0.8, 'soil_moisture': 27, 'temperature': 31, 'rainfall': 140
}

predicted_gwl = predict_groundwater(example_input)
print(f"\n💧 Predicted Groundwater Level: {predicted_gwl:.2f}")
