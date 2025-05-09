import streamlit as st
import cv2
import numpy as np
from src.style_transfer import load_image, apply_style
from src.super_resolution import upscale_image

st.title("AI-Powered Art Enhancer 🎨")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file:
    # Load and show original image
    image = cv2.imread(uploaded_file)
    st.image(image, caption="Original Image")

    # Apply NST
    style_img = load_image("data/style.jpg")  # Pre-defined artistic style
    stylized_image = apply_style(image, style_img)
    st.image(stylized_image, caption="Styled Image")

    # Apply Super-Resolution
    upscaled_image = upscale_image(uploaded_file)
    st.image(upscaled_image, caption="High-Resolution Output")
