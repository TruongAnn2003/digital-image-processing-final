import streamlit as st
import numpy as np
import cv2
from PIL import Image
import image_processing.image_processing as c3
import io
L = 256 

# Helper function to convert an OpenCV image to a format that can be displayed by Streamlit
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)
    return pil_img

# App title
st.title('Image Processing')

# Sidebar menu
st.sidebar.title("Menu")
menu_options = ["Open Image", "Open Image Color", "Save Image", "Negative", "Logarithm", "Power", "Piecewise Linear Gray", "Histogram", "Histogram Equalization","Blur","Edge Detection","Enhance Color"]
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

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

# Function to save the output image
def save_image():
    if st.session_state.imgout is not None:
        pil_img = cv2_to_pil(st.session_state.imgout)
        img_bytes = pil_to_bytes(pil_img)
        st.download_button(label="Download Image", data=img_bytes, file_name="output_image.png", mime="image/png")
        st.image(cv2_to_pil(st.session_state.imgout), caption="output_image.png", use_column_width=True)
    else:
        st.warning("Please process the image before saving.")

# Function to apply image processing operations
def process_image(operation):
    if st.session_state.imgin is not None:
        if operation == "Negative":
            st.session_state.imgout = c3.Negative(st.session_state.imgin)
        elif operation == "Logarithm":
            c_default = (L - 1) / np.log(1.0 * L)
            c = st.slider('Adjust the scaling constant c, default=%.2f' % c_default, min_value=0.1, max_value=100.0, value=c_default)
            st.session_state.imgout = c3.Logarit(st.session_state.imgin,c)
        elif operation == "Power":
            gamma_default = 5.0
            gamma = st.slider('Adjust the scaling constant gamma, default=%.2f' % gamma_default, min_value=0.1, max_value=10.0, value=gamma_default)
            st.session_state.imgout = c3.Power(st.session_state.imgin,gamma)
        elif operation == "Piecewise Linear Gray":
            st.session_state.imgout = c3.PiecewiseLinearGray(st.session_state.imgin)
        elif operation == "Histogram":
            st.session_state.imgout = c3.Histogram(st.session_state.imgin)
        elif operation == "Histogram Equalization":
            st.session_state.imgout = c3.HistEqual(st.session_state.imgin)
            original_hist = c3.Histogram(st.session_state.imgin)
            equalized_hist = c3.Histogram(st.session_state.imgout)
        elif operation == "Blur":
            blur_amount = st.slider('Blur Amount', min_value=1, max_value=100, step=2, value=1)
            st.session_state.imgout = cv2.GaussianBlur(st.session_state.imgin, (blur_amount, blur_amount), 0)
        elif operation == "Edge Detection":
            low_threshold = st.slider('Low Threshold', min_value=0, max_value=255, value=100)
            high_threshold = st.slider('High Threshold', min_value=0, max_value=255, value=200)
            st.session_state.imgout = cv2.Canny(st.session_state.imgin, low_threshold, high_threshold)
        elif operation =="Enhance Color":
            red_factor = st.slider("Red", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            green_factor = st.slider("Green", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            blue_factor = st.slider("Blue", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
            st.session_state.imgout = c3.EnhanceColor(st.session_state.imgin, red_factor, green_factor, blue_factor)
        
        st.image(cv2_to_pil(st.session_state.imgout), caption=operation, use_column_width=True)
        if  operation == "Histogram Equalization" :
            st.image(cv2_to_pil(original_hist), caption='Histogram Before', use_column_width=True)
            st.image(cv2_to_pil(equalized_hist), caption='Histogram After', use_column_width=True)
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
