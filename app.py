import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #ff0000;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction App")
st.write("Fill the details below to check heart disease risk")

# Layout in 2 columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate", 60, 220)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]],
                              columns=["age","sex","cp","trestbps","chol",
                                       "fbs","restecg","thalach","exang",
                                       "oldpeak","slope","ca","thal"])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.balloons()