import streamlit as st
import os
import sys

from tqdm import tqdm
import imageio
import numpy as np
import cv2
import torch
from torch import nn
from predict import run as predict
from PIL import Image


phone="iphone"
datadir="ref/test_img/" 
test_subset="full" 
resolution="orig" 
weight_path="models/100trainsize/weights/generator_epoches_209.pth"

def main():
    st.title("Image Enhancement")

    # Upload the image file
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        

        # Enhance the image
        enhanced_path = predict(phone, uploaded_file, test_subset, weight_path, resolution)

        # Display the original and enhanced images side by side
        image=Image.open(uploaded_file)
        enhanced=Image.open(enhanced_path)
        col1, col2 = st.beta_columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            st.subheader("Enhanced Image")
            st.image(enhanced, caption="Enhanced Image", use_column_width=True)

        # Add a download link for the enhanced image
        st.download_button("Download Enhanced Image", data=enhanced.tobytes(), file_name="enhanced_image.jpg")

if __name__ == "__main__":
    main()