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

The emotion recognition model used in this project can be trained using the Kaggle dataset. To train the model:

1. Go to the Kaggle dataset [Emotion Detection - FER](https://www.kaggle.com/ananthu017/emotion-detection-fer) page.

2. Download the dataset and place it in the `data` directory of this project.

3. Navigate to the `train_emotion_model.ipynb` notebook.

4. Follow the instructions in the notebook to train the emotion recognition model.

## Downloading the Pre-trained Emotion Recognition Model

You can download the pre-trained emotion recognition model from the following link:

[Download Emotion Recognition Model](https://www.kaggle.com/example/emotion-recognition-model)

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

## Acknowledgments

The emotion detection model used in this project is based on the following research paper:
[Facial Expression Recognition Using Convolutional Neural Networks: State of the Art](https://arxiv.org/abs/1612.02903)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
