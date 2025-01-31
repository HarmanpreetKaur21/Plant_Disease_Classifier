import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from groq import Groq

# Set page configuration
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

# Get the directory of the current file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Use os.path.join for cross-platform compatibility
model_path = os.path.join(working_dir, 'plant_disease_prediction_model.h5')
class_indices_path = os.path.join(working_dir, 'class_indices.json')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading the class names
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name  # Only returning the name of the disease

# Function to Get Solution from LLM
def get_solution_from_llm(disease_name):
    client = Groq(api_key="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")
    prompt = f"The plant disease detected is {disease_name}. Provide detailed steps to mitigate the disease and improve plant health."
    
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    solution = completion.choices[0].message.content
    return solution

# Sidebar Content
st.sidebar.title("🌿 About the App")
st.sidebar.write("This app uses AI to classify plant diseases and provide solutions.")
st.sidebar.subheader("How It Works:")
st.sidebar.write("""
1. Upload a clear image of a plant leaf.
2. The model predicts the disease using a pre-trained AI.
3. You get detailed mitigation solutions powered by LLMs.
""")
st.sidebar.markdown("**Technologies Used:**")
st.sidebar.write("- TensorFlow\n- PIL\n- Groq API\n- Streamlit")
st.sidebar.markdown("**Project By:** Harman")

# Main Page
st.markdown("<h1 style='text-align: center; color: green;'>🌱 Plant Disease Classifier</h1>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image (resized for smaller display)
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<h3 style='text-align: center;'>Uploaded Image</h3>", unsafe_allow_html=True)
        resized_img = image.resize((200, 200))  # Resized to 200x200 for smaller display
        st.image(resized_img, use_column_width=True)

    with col2:
        st.markdown("<h3 style='text-align: center;'>Prediction</h3>", unsafe_allow_html=True)
        if st.button("Classify"):
            with st.spinner("Analyzing the image..."):
                prediction = predict_image_class(model, uploaded_image, class_indices)  # Only name of the disease
                st.success(f"**Disease Detected:** {prediction}")

                st.markdown("---")
                st.markdown("<h3>Solution for the Problem:</h3>", unsafe_allow_html=True)
                solution = get_solution_from_llm(prediction)
                st.write(solution)
                
                # Toggle Solution Visibility
                with st.expander("Want to learn more? Click here!"):
                    st.markdown(solution)
                
                # Download Button
                st.download_button(
                    label="Download Solution as Text",
                    data=solution,
                    file_name=f"{prediction}_solution.txt",
                    mime="text/plain"
                )
else:
    st.warning("Please upload an image to proceed.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Made with ❤️ by Harman</p>",
    unsafe_allow_html=True
)
