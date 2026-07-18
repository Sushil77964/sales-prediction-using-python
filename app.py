import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Sales Prediction using Machine Learning",
    page_icon="📈",
    layout="centered"
)

# ---------------- Load Model ----------------
model = joblib.load("models/sales_prediction_model.pkl")

# ---------------- Title ----------------
st.title("📈 Sales Prediction using Machine Learning")
st.markdown(
    "Predict **Product Sales** based on **TV, Radio and Newspaper Advertisement Budget**."
)

st.divider()

# ---------------- User Inputs ----------------
st.subheader("📊 Enter Advertisement Budget")

tv = st.number_input(
    "TV Advertisement Budget",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
)

radio = st.number_input(
    "Radio Advertisement Budget",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
)

newspaper = st.number_input(
    "Newspaper Advertisement Budget",
    min_value=0.0,
    max_value=150.0,
    value=30.0,
)

st.divider()

# ---------------- Prediction ----------------
if st.button("🚀 Predict Sales", use_container_width=True):

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Sales: **{prediction[0]:.2f} Units**")

    st.balloons()

st.divider()

st.caption("Developed by Sushil Kumar ❤️")