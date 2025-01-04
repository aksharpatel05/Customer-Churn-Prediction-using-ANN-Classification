# Customer-Churn-Prediction-using-ANN-Classification

This project is an Artificial Neural Network (ANN) classification model implemented using TensorFlow and Streamlit. The model is designed to predict whether a customer will leave the bank service or not based on certain features.

## Project Structure

- `app.py`: The main application file that loads the model and performs data transformations, and sets up the Streamlit interface.
- `model.h5`: The trained ANN model.
- `lable_encoder_gender.pkl`: The label encoder for the gender feature.
- `onehotencoder_geography.pkl`: The one-hot encoder for the geography feature.
- `sc.pkl`: The standard scaler for feature scaling.

## Requirements

- Python 3.x
- TensorFlow
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the application.
