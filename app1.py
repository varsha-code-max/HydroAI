import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd

# -----------------------------
# 🎯 Title and description
# -----------------------------
st.set_page_config(page_title="💧 HydroAI - Groundwater Prediction", layout="centered")
st.title("💧 HydroAI: Predicting Groundwater Potential")
st.markdown("""
This AI-powered tool predicts **groundwater level or classification** based on soil and water parameters.  
Upload your soil readings or enter values below to get instant predictions.
""")

# -----------------------------
# 📦 Load model and scaler safely
# -----------------------------
model_path = "hydro_model.pkl"
scaler_path = "scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    st.success("✅ Model and scaler loaded successfully!")
else:
    st.error(
        f"""
        ❌ Model or scaler file not found!

        **Expected locations:**
        - {os.getcwd()}/{model_path}
        - {os.getcwd()}/{scaler_path}

        Please make sure both `.pkl` files are uploaded in the same folder as `app.py` (in your GitHub repo).
        """
    )
    st.stop()

# -----------------------------
# 🧮 Input features
# -----------------------------
st.header("🌾 Enter Soil & Water Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    ec = st.number_input("E.C (µS/cm)", min_value=0.0, max_value=10000.0, value=250.0)
    tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=500.0)

with col2:
    na = st.number_input("Sodium (Na)", min_value=0.0, max_value=500.0, value=50.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=10.0)
    ca = st.number_input("Calcium (Ca)", min_value=0.0, max_value=200.0, value=40.0)

with col3:
    mg = st.number_input("Magnesium (Mg)", min_value=0.0, max_value=200.0, value=30.0)
    so4 = st.number_input("Sulfate (SO₄)", min_value=0.0, max_value=500.0, value=100.0)
    cl = st.number_input("Chloride (Cl)", min_value=0.0, max_value=1000.0, value=150.0)

# -----------------------------
# 📊 Prepare input data
# -----------------------------
input_data = np.array([[ph, ec, tds, na, k, ca, mg, so4, cl]])
input_scaled = scaler.transform(input_data)

# -----------------------------
# 🔮 Prediction
# -----------------------------
if st.button("🔍 Predict Groundwater Potential"):
    try:
        prediction = model.predict(input_scaled)
        if isinstance(prediction[0], (np.integer, int)):
            st.success(f"💧 Predicted groundwater class: **{prediction[0]}**")
        else:
            st.success(f"💧 Prediction: **{prediction[0]}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("---")
st.caption("Developed with ❤️ by Sri Varsha | HydroAI Project | Edunet Initiative")
