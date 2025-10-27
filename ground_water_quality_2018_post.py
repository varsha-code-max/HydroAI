import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---- Dataset path ----
dataset_path = r"C:\Users\Sri Varsha\Desktop\edunet\ground_water_quality_2018_post.csv"

# ---- 1️⃣ Load dataset ----
df = pd.read_csv(dataset_path)
print("✅ Dataset loaded successfully!")
print(df.head())

# ---- 2️⃣ Define features and target ----
features = ['pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3 ', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC  meq  / L']
target = 'gwl'

# Drop rows with missing target values
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# ---- 3️⃣ Split data into train/test sets ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 4️⃣ Scale features ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- 5️⃣ Train Random Forest Regressor ----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- 6️⃣ Evaluate model ----
y_pred = model.predict(X_test)
print(f"Model Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Model R2 Score: {r2_score(y_test, y_pred):.2f}")

# ---- 7️⃣ Predict groundwater level from new input ----
def predict_groundwater(inputs_dict):
    """
    inputs_dict should contain keys for all features:
    'pH', 'E.C', 'TDS', 'CO3', 'HCO3', 'Cl', 'F', 'NO3 ', 'SO4', 'Na', 'K', 'Ca', 'Mg', 'T.H', 'SAR', 'RSC  meq  / L'
    Example:
    inputs_dict = {'pH': 7.2, 'E.C': 0.5, 'TDS': 300, ... }
    """
    data_input = [ [inputs_dict[f] for f in features] ]
    data_scaled = scaler.transform(data_input)
    prediction = model.predict(data_scaled)
    return prediction[0]

# ---- Example prediction ----
example_input = {
    'pH': 7.2, 'E.C': 0.5, 'TDS': 300, 'CO3': 10, 'HCO3': 150,
    'Cl': 50, 'F': 1, 'NO3 ': 20, 'SO4': 15, 'Na': 30, 'K': 5,
    'Ca': 40, 'Mg': 10, 'T.H': 200, 'SAR': 2, 'RSC  meq  / L': 1
}

predicted_gwl = predict_groundwater(example_input)
print(f"Predicted Groundwater Level (gwl): {predicted_gwl:.2f}")

