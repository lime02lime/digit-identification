import streamlit as st
import numpy as np
import requests
import cv2
import psycopg2
from streamlit_drawable_canvas import st_canvas
import os


# database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# API endpoint for the model container
MODEL_API_URL = "http://model-container:5000/predict" 

st.title("Handwritten Digit Recognition")

# Create two columns: one for the canvas and one for the outputs
col1, col2 = st.columns([1, 1])  # equal space for canvas and outputs

# Column 1: Drawing canvas
with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

# Column 2: Prediction results and input for actual number
with col2:
    # Input field for actual number
    true_label = st.text_input("Enter the actual number you drew:")

    if st.button("Submit"):
        if canvas_result.image_data is not None and true_label.strip() != "":
            # Convert to grayscale and resize to 28x28
            img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (28, 28))

            img = img.astype(np.uint8)  # Convert to uint8 for file sending
            
            # Convert to a temporary file-like object in memory
            _, img_encoded = cv2.imencode('.png', img)  # Encodes the image to PNG
            img_bytes = img_encoded.tobytes()  # Convert to bytes

            # Send the image as a file to model service
            response = requests.post(MODEL_API_URL, files={"image": ("image.png", img_bytes, "image/png")})
            
            # Load data from model
            data = response.json()
            prediction = data.get("prediction")
            confidence = data.get("confidence")

            st.write(f"#### Prediction: {prediction}")
            st.write(f"#### Confidence: {confidence:.3f}")
            st.write(f"#### True Label: {true_label}")

            try:
                # Insert the data to the db
                cursor.execute(
                    """
                    INSERT INTO submissions (true_label, prediction, confidence)
                    VALUES (%s, %s, %s)
                    """, (true_label, prediction, confidence)
                )
                conn.commit()  # Save changes to the database

                # Check if the data was inserted successfully
                cursor.execute(
                    """
                    SELECT true_label, prediction, confidence, timestamp FROM submissions WHERE true_label = %s AND prediction = %s AND confidence = %s ORDER BY timestamp DESC LIMIT 1
                    """, (true_label, prediction, confidence)
                )
                result = cursor.fetchone()  # Get the most recent matching row
                
                if result:
                    st.write(f"Data successfully inserted to db: {result}")
                else:
                    st.write("Error: Data not found in the database.")
            
            except psycopg2.Error as e:
                st.write(f"Error saving to db: {e}")
                conn.rollback()  # Rollback in case of an error

        else:
            st.error("Please draw a number AND enter the actual number before submitting!")