import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json


# Page Config

st.set_page_config(
    page_title="AI Garbage Classifier",
    page_icon="♻️",
    layout="wide"
)
 

# UI DESIGN

st.markdown("""
<style>
 
/* Background */
.main {
    background-color: #f5f7fb;
}
 
/* Hero Banner */
.banner {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
 
/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
 
/* Headings */
h1 {
    color: #1e3c72;
}
 
h2, h3 {
    color: #2a5298;
}
 
/* Buttons */
.stButton > button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
}
 
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}
 
</style>
""", unsafe_allow_html=True)
 

# HERO HEADER

st.markdown("""
<div class="banner">
    <h1>♻️ AI Garbage Classifier Dashboard</h1>
    <p>Smart Waste Segregation using AI | Clean Future 🌍</p>
</div>
""", unsafe_allow_html=True)
 

# Load Model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.h5")
 
model = load_model()
 

# Load Class Mapping

with open("class_indices.json") as f:
    class_indices = json.load(f)
 
class_names = list(class_indices.keys())
 

# Recycling Advice Function

def recycling_advice(label):
    if label in ["plastic", "glass", "metal"]:
        return "Put in recyclable bin"
    elif label in ["paper", "cardboard"]:
        return "Recycle or reuse"
    elif label in ["organic"]:
        return "Compostable waste"
    else:
        return "trash (Dispose of properly)"
 

# Sidebar Navigation

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "📷 Classifier", "📊 Insights", "📖 About"])
 

# HOME PAGE

if page == "🏠 Home":
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("♻️ Smart AI Garbage Classification System")
 
    st.write("""
    🚀 **Welcome to AI Garbage Classifier**
 
    This intelligent system uses Deep Learning to classify waste into:
 
    - Cardboard
    - Glass
    - Metal
    - Organic
    - Paper
    - Plastic
    - Trash
 
    ✅ Helps automate recycling 
    ✅ Reduces environmental impact 
    ✅ Supports smart city initiatives 
    """)
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 Why This Matters")
 
    st.write("""
    - Improper waste segregation harms the environment 
    - AI can automate and improve recycling efficiency 
    - Smart classification = cleaner future 
    """)
    st.markdown('</div>', unsafe_allow_html=True)
 

# CLASSIFIER PAGE

elif page == "📷 Classifier":
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("📷 Upload & Classify Waste")
 
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
 
    if uploaded_file:
        image = Image.open(uploaded_file)
 
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
        # Preprocess
        img = Image.open(uploaded_file).convert("RGB")  # Convert RGBA -> RGB
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
 
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        advice = recycling_advice(predicted_class)
 
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
 
            st.success(f"✅ Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}")
            st.warning(advice)
 
            st.markdown('</div>', unsafe_allow_html=True)
 
            # Chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.bar(class_names, prediction[0])
            ax.set_title("Prediction Confidence")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
 

# INSIGHTS PAGE

elif page == "📊 Insights":
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("📊 Model Insights")
 
    st.write("""
    🔍 This model learns visual features like:
 
    - Texture (glass vs plastic)
    - Shape (boxes vs bottles)
    - Color patterns
 
    📈 Accuracy improves with:
 
    - More data
    - Better image quality
    - Balanced datasets
    """)
    st.markdown('</div>', unsafe_allow_html=True)
 

# ABOUT PAGE

elif page == "📖 About":
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("📖 About This Project")
 
    st.write("""
    This project demonstrates:
 
    ✅ Deep Learning (CNN + MobileNetV2) 
    ✅ Image Classification 
    ✅ Real-world sustainability application 
 
    ⚙️ Built using:
 
    - Python 
    - TensorFlow 
    - Streamlit 
 
    🎯 Purpose:
 
    - Smart waste segregation 
    - Eco-friendly solutions 
    """)
    st.markdown('</div>', unsafe_allow_html=True)