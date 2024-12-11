import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define a function to load and preprocess the image
def preprocess_image(image):
    # Resize image to match model input size
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define a function to load the trained model
def load_trained_model(model_path):
    return load_model(model_path)

# UI layout
st.title("Chest X-ray Classification")
st.write("Upload a chest X-ray image and select a model to predict the condition.")

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

# Model selection
model_options = {
    "Model 1 - CNN Model": "best_model.h5",
    "Model 2 - Xception Model": "best_model.keras",
}
model_choice = st.selectbox("Select a Trained Model", list(model_options.keys()))

# Prediction button
if st.button("Predict"):
    if uploaded_file is not None and model_choice:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

        # Load and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        preprocessed_image = preprocess_image(image)

        # Load the selected model
        model_path = model_options[model_choice]
        model = load_trained_model(model_path)

        # Perform prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)

        # Display the prediction results
        st.write(f"Prediction: {predicted_class[0]}")
        st.write("Confidence Scores:", predictions[0])
    else:
        st.write("Please upload an image and select a model.")
