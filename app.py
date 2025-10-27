# app.py — Streamlit UI for HydroAI (safe model loading + graceful errors)
import streamlit as st
import numpy as np
import joblib
import os

# --- Path to where you saved model & scaler ---
BASE_DIR = r"C:\Users\Sri Varsha\Desktop\edunet"
MODEL_PATH = os.path.join(BASE_DIR, "hydro_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

st.set_page_config(page_title="HydroAI - Groundwater Predictor", page_icon="💧", layout="centered")
st.title("💧 HydroAI: Groundwater Potential Predictor")
st.write("Enter soil parameters to estimate groundwater availability potential.")

# ---- Load model & scaler safely ----
model = None
scaler = None
model_missing = False

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model_missing = True
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        model_missing = True
except Exception as e:
    model_missing = True
    st.error(f"Error loading model or scaler: {e}")

if model_missing:
    st.warning(
        "Model or scaler not found in:\n"
        f"  {MODEL_PATH}\n  {SCALER_PATH}\n\n"
        "Make sure you have trained the model and saved `hydro_model.pkl` and `scaler.pkl` in the folder above."
    )
    st.info("Tip: run your training script so those files are created, then refresh this page.")
    # still allow user to input values, but disable prediction
    allow_predict = False
else:
    allow_predict = True

# ---- UI inputs ----
col1, col2, col3 = st.columns(3)
with col1:
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
with col2:
    moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=1000.0, value=5.0, step=0.1)
with col3:
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=1000.0, value=10.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=-40.0, max_value=60.0, value=25.0, step=0.5)

st.write("")  # spacing

# ---- Determine expected feature order from scaler if available ----
# Default order used during training (keep consistent with your training script)
default_order = ['pH', 'moisture', 'nitrogen', 'phosphorus', 'potassium', 'temperature']

if scaler is not None and hasattr(scaler, "n_features_in_"):
    expected_n = int(scaler.n_features_in_)
    # Use the leading features in default_order
    feature_order = default_order[:expected_n]
else:
    feature_order = default_order

st.caption(f"Model expects features (in this order): {feature_order}")

# ---- Build input vector according to expected order ----
def build_input(order):
    mapping = {
        'pH': ph,
        'moisture': moisture,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'temperature': temperature
    }
    row = [mapping[f] for f in order]
    return np.array(row, dtype=float).reshape(1, -1)

# ---- Predict button (disabled when model/scaler missing) ----
predict_disabled = not allow_predict
if st.button("🔍 Predict Groundwater Potential", disabled=predict_disabled):
    try:
        X_input = build_input(feature_order)
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        # clamp between 0 and 100
        prediction_percent = float(np.clip(prediction, 0, 100))
        # show results
        st.success(f"💦 Estimated Groundwater Potential: **{prediction_percent:.2f}%**")
        if prediction_percent > 70:
            st.balloons()
            st.write("✅ Great potential for groundwater storage and farming!")
        elif prediction_percent > 40:
            st.write("⚠️ Moderate potential. Consider rainwater harvesting.")
        else:
            st.write("🚱 Low potential. Improve soil health or irrigation support.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("---")
st.caption("🌱 Empowering Sustainable Water Decisions- Created by Sri Varsha")
