import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import random
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json
from streamlit_drawable_canvas import st_canvas

OPTIMIZER = tf.keras.optimizers.Adam()

# Load model
model_architecture = 'handwritten_digit_recognition/digit_config.json'
model_weights = 'handwritten_digit_recognition/digit_weight.weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

# Load data
(_, _), (X_test, _) = datasets.mnist.load_data()
X_test = X_test.reshape((10000, 28, 28, 1))

def create_image_random():
    # Create 100 random numbers in the range [0, 9999]
    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100, 28, 28, 1))

    for i in range(100):
        sample[i] = X_test[index[i]]

    # Create image to view
    image = np.zeros((280, 280), dtype=np.uint8)
    k = 0
    for i in range(10):
        for j in range(10):
            image[i*28:(i+1)*28, j*28:(j+1)*28] = sample[k, :, :, 0]
            k += 1

    # Convert color and create PIL image
    color_converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(color_converted)
    image_tk = image_pil

    # Normalize
    data = sample / 255.0

    # Cast
    data = data.astype('float32')
    results = model.predict(data, verbose=0)

    digits = []
    for i in range(100):
        x = np.argmax(results[i])
        digits.append(x)

    return image_tk, digits

st.title("MNIST Digit Recognition (Random)")

# Button to create image
if st.button('Tạo ảnh và Nhận dạng'):
    image_tk, digits = create_image_random()
    st.image(image_tk, caption='Random Digits Image', use_column_width=True)
    s = ''
    for i in range(100):
        s += str(digits[i]) + " "
        if (i + 1) % 10 == 0:
            s += "\n"
    st.text(s)
    
# Function to preprocess the image and predict digit
def predict_digit(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 pixels
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize
    normalized = resized / 255.0
    # Reshape to match model input shape
    reshaped = normalized.reshape(1, 28, 28, 1)
    # Predict
    result = model.predict(reshaped)
    return np.argmax(result)

# Title of the app
st.title("MNIST Digit Recognition (Draw)")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Button to predict the drawn digit
if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Convert canvas image to array
        image_array = np.array(canvas_result.image_data, dtype=np.uint8)
        
        # Predict digit
        digit = predict_digit(image_array)
        st.write(f"Predicted digit: {digit}")
        # Display the drawn digit
        st.image(image_array, caption='Drawn Digit', use_column_width=True)
    else:
        st.write("Please draw a digit on the canvas!")
