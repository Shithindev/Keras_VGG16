import streamlit as st
import pickle
import cv2
import numpy as np
import tempfile

# Function to make predictions
def predict_mask(image, model):
    image_resized = cv2.resize(image, (224, 224))
    img = image_resized.reshape(1, 224, 224, 3)
    pred = model.predict(img)
    ind = np.argmax(pred)
    return ind

# Load the Keras model from the pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Mask Detection App")
st.write("Upload an image for mask detection.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image as bytes and convert to NumPy array
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict_mask(image, model)

    if prediction == 0:
        st.write("Prediction: Wearing mask")
    else:
        st.write("Prediction: Without mask")

