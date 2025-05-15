import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Solar Panel Defect Detector", layout="centered")

# Load model with caching
@st.cache_resource
def load_model_with_cache():
    try:
        return load_model('solar_panel_classifier.h5')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model_with_cache()

# Class names and maintenance messages
categories = ['Clean', 'Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']
maintenance_messages = {
    'Clean': "✅ No action needed. Panel is in perfect condition.",
    'Dusty': "🧹 Schedule cleaning soon to maintain efficiency.",
    'Bird-Drop': "🐦 Clean within a week to prevent damage.",
    'Electrical-Damage': "⚠️ Urgent! Contact a technician immediately.",
    'Physical-Damage': "🔧 Panel needs professional repair/replacement.",
    'Snow-Covered': "❄️ Clear snow when safe to restore power."
}
img_size = 128

# Streamlit UI
st.title("🔍 SolarGuard: Solar Panel Defect Detector")

st.markdown("""
Upload an image of a solar panel, and the model will:
1. Classify its condition
2. Provide maintenance recommendations
3. Show prediction confidence
""")

uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess with error handling
    try:
        img = np.array(image)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:  # RGBA to RGB
            img = img[:, :, :3]
        
        # Enhance image quality
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        st.stop()

    # Predict with spinner
    with st.spinner("Analyzing solar panel..."):
        prediction = model.predict(img)
        pred_index = np.argmax(prediction)
        confidence = np.max(prediction)
        condition = categories[pred_index]

    # Results in second column
    with col2:
        st.subheader("Analysis Results")
        
        # Condition with appropriate emoji
        st.metric("Condition", f"{condition} {maintenance_messages[condition].split()[0]}")
        
        # Confidence with progress bar
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(int(confidence * 100))
        
        # Maintenance recommendation
        st.subheader("Recommended Action")
        if condition in ['Electrical-Damage', 'Physical-Damage']:
            st.error(maintenance_messages[condition])
        elif condition in ['Dusty', 'Bird-Drop']:
            st.warning(maintenance_messages[condition])
        else:
            st.success(maintenance_messages[condition])
        
        # Show all probabilities
        with st.expander("Detailed probabilities"):
            for i, category in enumerate(categories):
                st.write(f"{category}: {prediction[0][i] * 100:.1f}%")

# Add footer
st.markdown("---")
st.caption("SolarGuard v1.0 | For maintenance inquiries, contact support@solarguard.ai")