*** This code is for the following dataset on Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset *** 


# Plant Disease Classifier

## Overview
This project aims to classify plant diseases from images using a Convolutional Neural Network (CNN) implemented in TensorFlow. The user interface is built using Streamlit.

## Steps to Run the Project

1. **Setup Google Colab with TPU:** Utilize Google Colab with a TPU (Tensor Processing Unit) for faster CNN training. 

2. **Download Dataset from Kaggle:** Obtain dataset credentials and the API command from Kaggle to download the dataset directly.

3. **Train the Model:** Follow the instructions in `ModelTrainPlantPrediction.ipynb` to train the CNN model.
    - Develop a simple CNN model with max-pooling and dropout features.
    - Build a predictive system for testing purposes.
    - Optionally, mount the trained model to Google Drive for faster access or copy it to local directories for Streamlit inference later.
    - Trained model available [here](https://drive.google.com/file/d/189ia_neoKtRrH7TsQrRfCxDIQ7ULkFzB/view?usp=drive_link).
    - Obtain the class indices for classification purposes.

4. **Run the Streamlit Interface:** Execute the `main.py` Python script to classify plant diseases from images using the pre-trained TensorFlow model and Streamlit.
    ```bash
    streamlit run main.py
    ```

5. **Dependencies:**
    - TensorFlow
    - Streamlit

6. **Troubleshooting:**
    - If encountering any issues, ensure all dependencies are installed correctly.
    - Check for any missing files or incorrect paths.



Credits: siddhardhan23 YT channel for the guidelines. 
