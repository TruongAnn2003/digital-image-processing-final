import streamlit as st
import numpy as np
import cv2
from PIL import Image
import image_processing.image_processing as c3

# Helper function to convert an OpenCV image to a format that can be displayed by Streamlit
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img

# App title
st.title('Image Processing')

# Sidebar menu
st.sidebar.title("Menu")
menu_options = ["Open Image", "Open Image Color", "Save Image", "Negative", "Negative Color", "Logarithm", "Power", "Piecewise Linear", "Histogram", "Histogram Equalization"]
selection = st.sidebar.radio("Choose an option", menu_options)

# State to hold images
if 'imgin' not in st.session_state:
    st.session_state.imgin = None

if 'imgout' not in st.session_state:
    st.session_state.imgout = None

# Function to handle image uploads
def upload_image(is_color):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif", "gif"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if is_color:
            st.session_state.imgin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.session_state.imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        st.image(cv2_to_pil(st.session_state.imgin), caption='Uploaded Image', use_column_width=True)

# Function to save the output image
def save_image():
    if st.session_state.imgout is not None:
        pil_img = cv2_to_pil(st.session_state.imgout)
        pil_img.save("output_image.png")
        st.success("Image saved as output_image.png")

# Function to apply image processing operations
def process_image(operation):
    if st.session_state.imgin is not None:
        if operation == "Negative":
            st.session_state.imgout = c3.Negative(st.session_state.imgin)
        elif operation == "Negative Color":
            st.session_state.imgout = c3.NegativeColor(st.session_state.imgin)
        elif operation == "Logarithm":
            st.session_state.imgout = c3.Logarit(st.session_state.imgin)
        elif operation == "Power":
            st.session_state.imgout = c3.Power(st.session_state.imgin)
        elif operation == "Piecewise Linear":
            st.session_state.imgout = c3.PiecewiseLinear(st.session_state.imgin)
        elif operation == "Histogram":
            st.session_state.imgout = c3.Histogram(st.session_state.imgin)
        elif operation == "Histogram Equalization":
            st.session_state.imgout = c3.HistEqual(st.session_state.imgin)
        st.image(cv2_to_pil(st.session_state.imgout), caption='Processed Image', use_column_width=True)
    else:
        st.warning("Please upload an image first")

# Menu options logic
if selection == "Open Image":
    upload_image(is_color=False)
elif selection == "Open Image Color":
    upload_image(is_color=True)
elif selection == "Save Image":
    save_image()
else:
    process_image(selection)
