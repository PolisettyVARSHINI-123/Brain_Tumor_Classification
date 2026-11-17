# Brain Tumor Classification Using CNN

## About the Project

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories. It includes a trained model and a Streamlit application for uploading MRI images and viewing predictions.

## Tumor Categories

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Technologies Used

- Python 3.12
- TensorFlow and Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Model Details

- Image Size: 224 × 224
- Batch Size: 32
- Epochs: 10
- Test Accuracy: 85.81%

## Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

## Disclaimer

This project is intended for educational and demonstration purposes only. It is not designed or validated for medical diagnosis.