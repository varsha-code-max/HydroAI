import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
dataset_path = r"C:\Users\Sri Varsha\Desktop\edunet"
file_path = os.path.join(dataset_path, "data_core.csv")

df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# -----------------------------
# 2️⃣ Create or Identify Target
# -----------------------------
if 'Groundwater_Level' not in df.columns:
    print("⚠️ 'Groundwater_Level' column not found.")
    print("👉 Creating a sample 'Groundwater_Level' column for testing.")
    import numpy as np
    # Simulated values for testing (you can replace with your real groundwater data)
    df['Groundwater_Level'] = (
        0.3 * df['Temparature'] 
        + 0.4 * df['Moisture'] 
        + 0.2 * df['Humidity'] 
        + np.random.uniform(0, 10, len(df))
    )

target = 'Groundwater_Level'

# -----------------------------
# 3️⃣ Encode Categorical Columns
# -----------------------------
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -----------------------------
# 4️⃣ Select Features and Target
# -----------------------------
features = ['Temparature', 'Humidity', 'Moisture', 
            'Soil Type', 'Crop Type', 'Nitrogen', 
            'Potassium', 'Phosphorous', 'Fertilizer Name']

X = df[features]
y = df[target]

# -----------------------------
# 5️⃣ Split and Scale
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6️⃣ Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 7️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# -----------------------------
# 8️⃣ Prediction Function
# -----------------------------
def predict_groundwater(temparature, humidity, moisture, soil_type, crop_type,
                        nitrogen, potassium, phosphorous, fertilizer_name):

    # Encode categorical inputs
    soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
    crop_encoded = label_encoders['Crop Type'].transform([crop_type])[0]
    fert_encoded = label_encoders['Fertilizer Name'].transform([fertilizer_name])[0]

    input_data = pd.DataFrame([[
        temparature, humidity, moisture, soil_encoded, crop_encoded,
        nitrogen, potassium, phosphorous, fert_encoded
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return prediction

# -----------------------------
# 🔬 Example Test Prediction
# -----------------------------
pred = predict_groundwater(
    temparature=30,
    humidity=60,
    moisture=40,
    soil_type='Sandy',
    crop_type='Maize',
    nitrogen=30,
    potassium=20,
    phosphorous=10,
    fertilizer_name='Urea'
)
print(f"\n💧 Predicted Groundwater Level: {pred:.2f}")
