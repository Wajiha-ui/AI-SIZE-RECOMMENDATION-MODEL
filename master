import streamlit as st
import numpy as np
from catboost import CatBoostClassifier

# Load the CatBoost model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("size_recommender.cbm")  # Load .cbm model
    return model

model = load_model()

st.title("Nyx Size Recommender")
st.write("Enter your details to get a recommended size.")

# User inputs
height = st.number_input("Enter height (cm):", min_value=140, max_value=220)
weight = st.number_input("Enter weight (kg):", min_value=30, max_value=150)

if st.button("Get Recommendation"):
    input_data = np.array([[height, weight]])
    prediction = model.predict(input_data)
    st.success(f"Recommended Size: {prediction[0]}")
