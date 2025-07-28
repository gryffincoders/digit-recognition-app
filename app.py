
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("mnist_cnn_model.h5")

st.title("🖌️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and click 'Predict'.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = canvas_result.image_data
    img = Image.fromarray((image[:, :, 0:1] * 255).astype(np.uint8).squeeze())
    img = ImageOps.invert(img).resize((28, 28)).convert('L')
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0

    if st.button("Predict"):
        prediction = model.predict(img)
        st.success(f"Predicted Digit: **{np.argmax(prediction)}**")
