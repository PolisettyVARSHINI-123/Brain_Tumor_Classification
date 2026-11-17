import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#image parameters
image_size = 224
Batch_size = 32
Epochs = 5

# Data set paths
train_path = r"D:\Medical_Image_Classification\dataset\Training"
test_path = r"D:\Medical_Image_Classification\dataset\Testing"

#image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=Batch_size,
    class_mode='categorical',
    subset='training'
)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=Batch_size,
    class_mode='categorical',
    shuffle=False
)
validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=Batch_size,
    class_mode='categorical',
    subset='validation'
)

# Convolution layer process CNN model
model = Sequential()

#1st CN Layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#2nd Convolution Layer
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#3rd Convolution Layer
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten Layer
model.add(Flatten())

#fully connected layer
model.add(Dense(128,activation='relu'))

#Dropout layer
model.add(Dropout(0.5))

#output layer
model.add(Dense(4,activation='softmax'))

#Display Model Architecture
model.summary()

#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("Model compiled Successfully")

#train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=Epochs
)

# Training and Validation Accuracy Graph

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png")
plt.show()

# Training and Validation Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("loss_graph.png")

plt.show()


# Evaluate Model

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy * 100, "%")

# Save Model

model.save("brain_tumor_model.keras")

print("Model saved successfully!")

# Save Class Names

import json

class_names = list(train_generator.class_indices.keys())

with open("class_names.json", "w") as file:
    json.dump(class_names, file)

print("Class names saved successfully!")


