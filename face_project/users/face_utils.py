import cv2
import mediapipe as mp
import pickle
import os
import numpy as np
import face_recognition
from fer import FER

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FACE_DATA_FILE = "faces.pkl"
MIN_FACE_SIZE = 50
TARGET_FACE_SIZE = 160
MAX_FRAME_SIZE = 800

# Load once (avoid re-init every call)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
expression_detector = FER(mtcnn=True)  # Changed to True for better accuracy

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def detect_faces(frame):
    """Detect faces in a frame using MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    face_regions = []
    if results.detections:
        h, w = frame.shape[:2]
        for detection in results.detections:
            if detection.score[0] < 0.6:
                continue

            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Expand the bounding box slightly to capture more facial features
            expansion = 0.15
            x_exp = max(0, int(x - width * expansion))
            y_exp = max(0, int(y - height * expansion))
            width_exp = min(int(width * (1 + 2 * expansion)), w - x_exp)
            height_exp = min(int(height * (1 + 2 * expansion)), h - y_exp)

            if width_exp >= MIN_FACE_SIZE and height_exp >= MIN_FACE_SIZE:
                face_roi = frame[y_exp:y_exp+height_exp, x_exp:x_exp+width_exp]
                if face_roi.size > 0:
                    face_regions.append(face_roi)

    return face_regions


def enhance_face(face_img):
    """Enhance face image for better feature extraction"""
    try:
        # Convert to RGB first
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize if too small
        h, w = rgb_face.shape[:2]
        if min(h, w) < TARGET_FACE_SIZE:
            scale = TARGET_FACE_SIZE / float(min(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            rgb_face = cv2.resize(rgb_face, (new_w, new_h))
        
        # Apply histogram equalization to improve contrast
        ycrcb = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing face: {e}")
        return None


def get_encoding(face_img):
    """Get face encoding for recognition"""
    enhanced = enhance_face(face_img)
    if enhanced is None:
        return None
        
    # Convert to RGB (face_recognition expects RGB)
    rgb_face = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if len(enhanced.shape) == 3 and enhanced.shape[2] == 3 else enhanced
    
    # Try multiple times with different parameters if needed
    encodings = face_recognition.face_encodings(rgb_face)
    if encodings:
        return encodings[0]
    
    # If no encoding found, try with the original image
    rgb_original = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_original)
    return encodings[0] if encodings else None


def detect_emotion(face_img):
    """Detect emotion from face image using FER"""
    try:
        # Resize to a standard size for FER
        resized = cv2.resize(face_img, (48, 48))
        
        # Convert to RGB (FER expects RGB)
        rgb_face = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Detect emotions
        emotions = expression_detector.detect_emotions(rgb_face)
        
        if emotions and len(emotions) > 0:
            emotion_dict = emotions[0]["emotions"]
            # Get the emotion with the highest score
            emotion, score = max(emotion_dict.items(), key=lambda x: x[1])
            return emotion.capitalize(), float(score)
        
        return "Neutral", 0.5
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "Neutral", 0.5


def load_faces():
    """Load face data from file"""
    if os.path.exists(FACE_DATA_FILE):
        try:
            with open(FACE_DATA_FILE, "rb") as f:
                data = pickle.load(f)
                # Ensure the data has the correct structure
                if "encodings" not in data:
                    data = {"encodings": [], "names": []}
                return data
        except Exception as e:
            print(f"Error loading face data: {e}")
            return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}


def save_faces(data):
    """Save face data to file"""
    # Ensure the data has the correct structure
    if "encodings" not in data:
        data = {"encodings": [], "names": []}
        
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)


# Alternative emotion detection using facial features (fallback)
def detect_emotion_alternative(face_img):
    """Alternative emotion detection using facial feature analysis"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Get face dimensions
        height, width = gray.shape
        
        # Divide face into regions
        top_half = gray[0:height//2, :]
        bottom_half = gray[height//2:, :]
        
        # Calculate intensity histograms
        top_hist = cv2.calcHist([top_half], [0], None, [256], [0, 256])
        bottom_hist = cv2.calcHist([bottom_half], [0], None, [256], [0, 256])
        
        # Normalize histograms
        top_hist = cv2.normalize(top_hist, top_hist).flatten()
        bottom_hist = cv2.normalize(bottom_hist, bottom_hist).flatten()
        
        # Calculate histogram correlation
        correlation = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_CORREL)
        
        # Calculate standard deviation as a measure of contrast
        std_dev = np.std(gray)
        
        # Simple heuristic-based emotion detection
        if std_dev > 60:
            if correlation < 0.4:
                return "Surprise", 0.8
            else:
                return "Happy", 0.7
        elif std_dev < 40:
            if correlation > 0.7:
                return "Sad", 0.7
            else:
                return "Neutral", 0.6
        else:
            if correlation < 0.5:
                return "Angry", 0.7
            else:
                return "Neutral", 0.6
                
    except Exception as e:
        print(f"Error in alternative emotion detection: {e}")
        return "Neutral", 0.5