import cv2
import numpy as np
from PIL import Image

def analyze_environment(image_file):
    """
    Analyzes environment context (no face detection).
    Returns environment_state and risk_weight.
    """

    image = Image.open(image_file).convert("RGB")
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Brightness analysis
    brightness = np.mean(gray)

    # Edge density (rough clutter indicator)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # -------- Context Rules --------
    if brightness < 70:
        environment = "Dark / Low-energy environment"
        risk_weight = 2
    elif edge_density > 0.12:
        environment = "Cluttered / Busy environment"
        risk_weight = 1
    else:
        environment = "Normal / Stable environment"
        risk_weight = 0

    return environment, risk_weight
