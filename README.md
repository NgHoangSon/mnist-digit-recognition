# MNIST Digit Recognition

A handwritten digit recognition application built with PyTorch and Streamlit.

This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from 0 to 9.

## Features

- Upload handwritten digit images (PNG, JPG, JPEG).
- Preprocess images to match the MNIST format (28x28 grayscale).
- Predict handwritten digits from 0 to 9 using a trained CNN model.
- Display prediction confidence.
- Visualize probability distribution across all 10 digit classes.
- Display the preprocessed image for inspection.

## Project Structure

2.MNIST/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── mnist_cnn_best.pth
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── preprocessing.py
│   └── inference.py
│
├── my_images/
│   └── digit3.png
│
└── data/

## Installation

### 1. Clone the repository
git clone <your-repository-url>
cd 2.MNIST

### 2.Create a virtual environment
conda create -n mnist python=3.14
conda activate mnist

### 3 Install dependencies
pip install -r requirements.txt

## Usage
python -m streamlit run app.py

## Model Information

- **Architecture:** Convolutional Neural Network (CNN)
- **Dataset:** MNIST
- **Input shape:** (1, 28, 28)
- **Output classes:** 10 (digits 0-9)
- **Framework:** PyTorch
- **Model weights:** `model/mnist_cnn_best.pth`
- **Test accuracy:** 99.07%

## Example

The application allows users to upload a handwritten digit image and displays:

- The uploaded image.
- The preprocessed 28x28 image.
- The predicted digit.
- Prediction confidence.
- Probability distribution across all 10 classes.