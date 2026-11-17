import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load trained CNN model

model = tf.keras.models.load_model(
    r"dataset\brain_tumor_model.keras"
)

print("Model loaded successfully!")

# Load class names

with open(r"dataset\class_names.json", "r") as file:
    class_names = json.load(file)

    # Load MRI image

    img = Image.open(r"D:\Medical_Image_Classification\sample.jpg.jpg").convert("RGB")

    # Resize image

    img = img.resize((224, 224))

    # Convert image into array

    img_array = np.array(img)

    # Normalize pixel values

    img_array = img_array / 255.0

    # Expand dimensions

    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction

prediction = model.predict(img_array)

# Get predicted class

result = np.argmax(prediction)

print("Predicted Class:", class_names[result])

print("Confidence:", prediction[0][result] * 100, "%")
