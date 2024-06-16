# Recommend Songs Based on Emotion

This project is an application that performs real-time emotion detection using computer vision and recommends songs based on the detected emotions. It utilizes OpenCV for face detection and a pre-trained deep learning model for emotion recognition. The recommended songs are fetched from the Spotify API.

## Installation

To run this application, you need to follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/hoduy511/recommend-songs-based-on-emotion.git
    ```

2. Navigate to the project directory:
    ```bash
    cd recommend-songs-based-on-emotion
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run emotion_detection.py
    ```

## Training the Emotion Recognition Model

The emotion recognition model used in this project can be trained using the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). To train the model:

1. Go to the Kaggle notebook [Train Emotions](https://www.kaggle.com/code/ngcduyh/train-emotions).

2. Follow the instructions in the notebook to train the emotion recognition model.

## Downloading the Pre-trained Emotion Recognition Model

You can download the pre-trained emotion recognition model from the following link:

[Download Emotion Recognition Model](https://www.kaggle.com/code/ngcduyh/train-emotions/output)

## Usage

1. When you run the application, it will open a web interface in your default browser.

2. The webcam feed will be displayed on the left side of the interface, showing real-time emotion detection results.

3. The recommended songs based on the detected emotion will be displayed on the right side of the interface.

4. Enjoy the recommended songs based on your emotions!

## Requirements

- Python 3.6+
- OpenCV
- TensorFlow
- Spotipy
- Streamlit
