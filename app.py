import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="Gender Classification", layout="centered")
st.title("👤 Gender Classification")
st.markdown("Upload a face image, and the model will predict if it's **Male** or **Female**.")

# Load the model (HDF5 format)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()
img_height, img_width = 200, 200

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image = image.resize((img_width, img_height))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        confidence = round(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100, 2)
        label = "Male" if prediction > 0.5 else "Female"

        # Display results
        st.markdown(f"### 🧠 Prediction: `{label}`")
        st.markdown(f"**Confidence:** `{confidence}%`")

    except Exception as e:
        st.error(f"Something went wrong while processing the image: {e}")
