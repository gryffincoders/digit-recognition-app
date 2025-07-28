import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognition")
uploaded_file = st.file_uploader("Upload a 28x28 digit image")

if uploaded_file:
    model = tf.keras.models.load_model("handwritten_model.h5")

    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")

