import streamlit as st
import numpy as np
import requests
import cv2
import json
from streamlit_drawable_canvas import st_canvas

# Define API endpoint for the model container
MODEL_API_URL = "http://model-container:5000/predict"  # Update with actual container URL

st.title("Handwritten Digit Recognition")

# Create a 28x28 drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
    )

# Input field for actual number
actual_number = st.text_input("Enter the actual number you drew:")

if st.button("Submit"):
    if canvas_result.image_data is not None and actual_number is not None:
        # Convert to grayscale and resize to 28x28
        img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalize
        img = img.astype(np.uint8)  # Convert to uint8 for file sending
        
        # Convert to a temporary file-like object in memory
        _, img_encoded = cv2.imencode('.png', img)  # Encodes the image to PNG
        img_bytes = img_encoded.tobytes()  # Convert to bytes

        # Send the image as a file to model service
        response = requests.post(MODEL_API_URL, files={"image": ("image.png", img_bytes, "image/png")})
        
        # load data from model
        #data = response.json()
        st.write(f"### Model Prediction: {response.text}")
        
        """
        prediction = data.get("prediction")
        probability = data.get("probability")
        
        st.write(f"### Model Prediction: {prediction}")
        st.write(f"### Prediction certainty: {probability:.3f}")
        st.write(f"### Actual Number: {actual_number}")"
        """
    else:
        st.error("Please draw a number AND enter the actual number before submitting!")
