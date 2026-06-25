
import streamlit as st
import joblib 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


st.title("Handwritten digits")

uploaded_file = st.file_uploader("Incarcati un fisier de tip 8x8 pixel", ["png", "jpg", "jpeg"])

model = joblib.load("digits_model.gz")


if uploaded_file:
    print(type(uploaded_file))
    np_image = np.array(Image.open(uploaded_file).convert("L")) / 16
    st.image(np_image / 16)
    predicted_value = model.predict([np_image.flatten()])
    st.write("Valoare prezisa este:", predicted_value[0])

    