import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Handwritten Digit Recognition")
uploaded_file = st.file_uploader("Upload a 28x28 digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load model only once
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("handwritten_model.h5")

    model = load_model()

    # Process image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)                  # Invert colors
    image = image.resize((28, 28))                  # Resize to 28x28

    # Convert to numpy and normalize
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)  # ✅ FIXED SHAPE

    st.image(image, caption="Uploaded Digit", width=150)

    # Prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {predicted_digit}")



