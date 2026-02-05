from fer import FER
import cv2
import numpy as np
from PIL import Image

# Initialize FER detector
emotion_detector = FER(mtcnn=True)

def analyze_facial_emotion(image_file):
    """
    Detects facial emotion from image.
    Returns: (emotion_label, emotion_score, stress_weight)
    """

    image = Image.open(image_file).convert("RGB")
    img_array = np.array(image)

    results = emotion_detector.detect_emotions(img_array)

    if not results:
        return "no_face_detected", 0.0, 0

    emotions = results[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    score = emotions[top_emotion]

    # Map emotion → stress weight
    if top_emotion in ["angry", "sad", "fear"]:
        stress_weight = 3
    elif top_emotion == "neutral":
        stress_weight = 1
    else:  # happy
        stress_weight = 0

    return top_emotion, score, stress_weight
