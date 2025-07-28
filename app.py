import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognition")

@st.cache_resource
def load_model():
    try:
       tf.keras.models.load_model("handwritten_model.keras")

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

uploaded_file = st.file_uploader("Upload a 28x28 digit image (in .png or .jpg format)")

if uploaded_file:
    model = load_model()
    if model is None:
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28)

        prediction = model.predict(image_array)
        st.success(f"Predicted Digit: {np.argmax(prediction)}")

    except Exception as e:
        st.error(f"Failed to process image or predict: {e}")



