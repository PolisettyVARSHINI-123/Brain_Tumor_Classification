
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# CNN model train
model = tf.keras.models.load_model(
    r"D:\Medical_Image_Classification\dataset\brain_tumor_model.keras"
)
print("Model Loaded!")



test_path = r"D:\Medical_Image_Classification\dataset\Testing"

# img parameters
image_size = 224
batch_size = 32

# test imgs
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("Testing images loaded successfully!")

# model test imgs
test_loss, test_accuracy = model.evaluate(test_generator)

print("\nModel Evaluation Results")
print("------------------------")

print("Test Loss:", round(test_loss, 4))

print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# Predict test imgs
predictions = model.predict(test_generator)

# pred cls labels
y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes

print("\nPredictions completed successfully!")
class_names = list(test_generator.class_indices.keys())
print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Brain Tumor Classification - Confusion Matrix")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# Plot confusion matrix  heatmap
plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()

# confusion matrix  image
plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Brain Tumor Classification")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(r"D:\Medical_Image_Classification\dataset\confusion_matrix.png.",
    dpi=300,
    bbox_inches="tight"
    )
print("Confusion matrix saved successfully!")
plt.show()


#results
with open(
    r"D:\Medical_Image_Classification\dataset\evaluation_results.txt",
    "w"
) as file:

    file.write("Medical Image Classification Results\n")

    file.write("------------------------------------\n")

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write("\nClassification Report:\n")

    file.write(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

print("Evaluation results saved successfully!")

