import cv2
import numpy as np
import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize emotion detection model
emotion_model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax"),
    ]
)

# Load pre-trained weights for the emotion model
emotion_model.load_weights("model.weights.h5")

# Dictionary to map emotion indices to corresponding emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Set up Spotify authentication
client_credentials_manager = SpotifyClientCredentials(
    client_id="46cd12a21552448dbf4ebeb0eba75657",
    client_secret="5243b4da10c34db5b60d5e977b929cab",
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# Function to predict emotion from frame
def predict_emotion(frame, model):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    emotion_label = None

    # Predict emotion for each detected face
    for x, y, w, h in faces:
        # Extract ROI of face
        roi_gray = gray_frame[y : y + h, x : x + w]
        # Resize face ROI to 48x48 pixels (model input size)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        # Reshape face ROI to match model input size
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        # Normalize pixel values
        roi_gray = roi_gray / 255.0
        # Predict emotion
        predictions = model.predict(roi_gray)
        # Get index of predicted emotion
        emotion_idx = np.argmax(predictions)
        # Get corresponding emotion label
        emotion_label = emotion_dict[emotion_idx]
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the predicted emotion label above the face box
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame, emotion_label


# Function to recommend songs based on emotion
def recommend_songs(emotion):
    if emotion is None:
        return []

    # Map emotion to Spotify search keyword
    emotion_to_music = {
        "Angry": "angry",
        "Disgusted": "disgust",
        "Fearful": "fear",
        "Happy": "happy",
        "Neutral": "calm",
        "Sad": "sad",
        "Surprised": "surprise",
    }

    query = emotion_to_music.get(emotion, "happy")

    # Search for songs on Spotify
    results = sp.search(q=f"track:{query}", type="track", limit=10)
    tracks = results["tracks"]["items"]

    # Extract song information
    song_recommendations = []
    for track in tracks:
        song_recommendations.append(
            {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "url": track["external_urls"]["spotify"],
            }
        )

    return song_recommendations


# Main function to run realtime emotion detection application
def main():
    # Set title for the application
    st.title("Realtime Emotion Detection and Music Recommendation")

    # Create two columns
    col1, col2 = st.columns(2)

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Set lower resolution to reduce load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Placeholder for displaying video and emotion detection results
    frame_container = col1.empty()
    songs_container = col2.empty()

    emotion = None

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Predict emotion from frame
        frame, new_emotion = predict_emotion(frame, emotion_model)

        # Display frame with predicted emotion label
        frame_container.image(
            frame, channels="BGR", use_column_width=True, caption="Camera"
        )

        # Update song list only if emotion changes
        if new_emotion and new_emotion != emotion:
            emotion = new_emotion
            songs = recommend_songs(emotion)
            if songs:
                for song in songs:
                    songs_container.write(
                        f"{song['name']} by {song['artist']} - [Listen on Spotify]({song['url']})"
                    )
            else:
                songs_container.write(f"No songs found for {emotion} mood.")

    # Release webcam
    cap.release()


# Run Streamlit application
if __name__ == "__main__":
    main()
