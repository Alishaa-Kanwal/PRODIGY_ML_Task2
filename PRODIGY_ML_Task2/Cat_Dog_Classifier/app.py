import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import time

# Load the saved model
model = keras.models.load_model('model/cat_dog_classification.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((256, 256))  # Resize image to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit app layout
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶🐱", layout="centered")
st.title("🐶 Cat vs Dog Classification 🐱")
st.write("Upload an image of a cat or dog and click 'Predict' to see the result.")

# File uploader widget to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, clamp=True)

    # Add a predict button
    if st.button("Predict"):
        # Show a status message while the prediction is being processed
        with st.spinner('Analyzing...'):
            time.sleep(2)  # Simulating a delay for the prediction

        # Show a progress bar to enhance the user experience
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)  # Simulating some computation delay
            progress_bar.progress(percent_complete + 1)

        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        # Display prediction results
        result_text = "Prediction: Dog" if prediction[0] > 0.5 else "Prediction: Cat"
        result_color = "red" if prediction[0] > 0.5 else "green"
        
        # Add an icon representing the prediction
        icon = "🐶" if prediction[0] > 0.5 else "🐱"

        # Display the result with confidence and an icon
        st.markdown(f"<h2 style='color: {result_color};'>{icon} {result_text} ({confidence*100:.2f}% confidence)</h2>", unsafe_allow_html=True)

# Add custom styling for the app
st.markdown("""
    <style>
        .stProgress > div {
            background-color:rgba(200, 18, 18, 0.9);
            border-radius: 5px;
        }
        h2 {
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Include custom JavaScript for smooth transitions (optional)
st.markdown('<script src="static/script.js"></script>', unsafe_allow_html=True)
