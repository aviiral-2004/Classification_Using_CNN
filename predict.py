import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("catdog_model.h5")

# Function to make predictions
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize like training

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Return prediction as text
    return "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
