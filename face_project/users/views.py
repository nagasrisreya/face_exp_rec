from django.shortcuts import render
from django.http import JsonResponse
import cv2
import mediapipe as mp
import pickle
import os
import base64
import numpy as np
import face_recognition
import math

# ------------------------------------------------------------------
# Config and Initialization
# ------------------------------------------------------------------
FACE_DATA_FILE = "faces.pkl"

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Confidence thresholds
EMOTION_CONFIDENCE = 0.4

# Minimum face size for processing
MIN_FACE_SIZE = 100  # pixels

# ------------------------------------------------------------------
# Enhanced Emotion Detection using Facial Landmarks
# ------------------------------------------------------------------
def detect_emotion_from_landmarks(face_img):
    """Detect emotion using facial landmarks and geometric features"""
    try:
        if face_img.size == 0:
            return "Neutral", 0.5
            
        # Convert to RGB for MediaPipe
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        results = face_mesh.process(rgb_face)
        
        if not results.multi_face_landmarks:
            return "Neutral", 0.5
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = face_img.shape[:2]
        
        # Get key landmark points
        try:
            # Mouth points (for smile detection)
            lip_top = landmarks[13]    # Upper lip
            lip_bottom = landmarks[14] # Lower lip
            mouth_left = landmarks[61] # Left corner
            mouth_right = landmarks[291] # Right corner
            
            # Eye points (for surprise)
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            # Eyebrow points (for anger/surprise)
            left_eyebrow = landmarks[65]
            right_eyebrow = landmarks[295]
            
            # Calculate features
            mouth_height = abs(lip_bottom.y * h - lip_top.y * h)
            mouth_width = abs(mouth_right.x * w - mouth_left.x * w)
            mouth_aspect_ratio = mouth_height / (mouth_width + 1e-5)
            
            left_eye_openness = abs(left_eye_bottom.y * h - left_eye_top.y * h)
            right_eye_openness = abs(right_eye_bottom.y * h - right_eye_top.y * h)
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
            
            # Emotion detection logic
            emotions = []
            confidences = []
            
            # Happy - wide smile (high mouth aspect ratio)
            if mouth_aspect_ratio > 0.15:
                emotions.append("Happy")
                confidences.append(min(0.8, mouth_aspect_ratio * 3))
            
            # Surprised - wide open eyes
            if avg_eye_openness > 15:
                emotions.append("Surprise")
                confidences.append(min(0.8, avg_eye_openness / 25))
            
            # Sad - downturned mouth corners (simplified)
            if mouth_aspect_ratio < 0.01:
                emotions.append("Sad")
                confidences.append(0.6)
            
            # If no strong emotions detected, return neutral
            if not emotions:
                return "Neutral", 0.6
                
            # Return the emotion with highest confidence
            best_idx = np.argmax(confidences)
            best_confidence = confidences[best_idx]
            
            if best_confidence >= EMOTION_CONFIDENCE:
                return emotions[best_idx], best_confidence
            else:
                return "Neutral", best_confidence
                
        except IndexError:
            # Fallback if landmark indices are out of range
            return "Neutral", 0.5
            
    except Exception as e:
        print(f"Landmark emotion detection error: {e}")
        return "Neutral", 0.5

def detect_emotion_with_opencv(face_img):
    """Alternative emotion detection using OpenCV features"""
    try:
        if face_img.size == 0:
            return "Neutral", 0.5
            
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect smiles using Haar cascade (simple approach)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if smile_cascade.empty():
            return detect_emotion_from_landmarks(face_img)
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        if len(smiles) > 0:
            return "Happy", 0.7
        
        # If no smile detected, use landmark-based approach
        return detect_emotion_from_landmarks(face_img)
        
    except Exception as e:
        print(f"OpenCV emotion detection error: {e}")
        return "Neutral", 0.5

# ------------------------------------------------------------------
# Enhanced Face Processing Functions
# ------------------------------------------------------------------
def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe with better quality control"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    face_locations = []
    face_regions = []
    
    if results.detections:
        for detection in results.detections:
            if detection.score[0] < 0.6:
                continue
                
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                # Add padding
                padding = int(min(width, height) * 0.15)
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(width + 2 * padding, w - x)
                height = min(height + 2 * padding, h - y)
                
                face_locations.append((y, x + width, y + height, x))
                face_region = frame[y:y+height, x:x+width]
                
                if face_region.size > 0:
                    face_regions.append(face_region)
    
    return face_locations, face_regions

def enhance_face_image(face_img):
    """Enhance face image for better feature extraction"""
    try:
        if face_img.size == 0:
            return None
            
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize for consistency
        target_size = 300
        h, w = rgb_face.shape[:2]
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb_face = cv2.resize(rgb_face, (new_w, new_h))
        
        # Improve contrast
        if len(rgb_face.shape) == 3:
            ycrcb = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            rgb_face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        return rgb_face
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return None

def get_face_encoding(face_img):
    """Extract face encoding with better error handling"""
    try:
        enhanced_face = enhance_face_image(face_img)
        if enhanced_face is None:
            return None
        
        encodings = face_recognition.face_encodings(enhanced_face)
        if encodings:
            return encodings[0]
        else:
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            return encodings[0] if encodings else None
            
    except Exception as e:
        print(f"Face encoding error: {e}")
        return None

# ------------------------------------------------------------------
# Persistence Helpers
# ------------------------------------------------------------------
def load_faces():
    if os.path.exists(FACE_DATA_FILE):
        try:
            with open(FACE_DATA_FILE, "rb") as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    return {"encodings": [], "names": []}
                return {
                    "encodings": list(data.get("encodings", [])),
                    "names": list(data.get("names", [])),
                }
        except:
            return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}

def save_faces(data):
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

# ------------------------------------------------------------------
# Views
# ------------------------------------------------------------------
def home(request):
    return render(request, "home.html")

def register_user(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        image_data = request.POST.get("image")

        if not name or not image_data:
            return JsonResponse({"message": "Name and image are required!"})

        try:
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({"message": "Invalid image data."})

            # Resize if too large
            h, w = frame.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Detect faces
            face_locations, face_regions = detect_faces_mediapipe(frame)
            
            if not face_locations:
                return JsonResponse({"message": "No face detected. Try again with better lighting."})

            face_roi = face_regions[0]
            encoding = get_face_encoding(face_roi)
            
            if encoding is None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
                if face_encodings:
                    encoding = face_encodings[0]
                else:
                    return JsonResponse({"message": "Could not extract facial features."})

            # Check for duplicates
            data = load_faces()
            if data["encodings"]:
                dists = face_recognition.face_distance(data["encodings"], encoding)
                best_i = int(np.argmin(dists))
                best_d = float(dists[best_i])
                if best_d < 0.4 and data["names"][best_i].lower() == name.lower():
                    return JsonResponse({"message": f"{name} is already registered."})
            
            # Save face data
            data["encodings"].append(encoding)
            data["names"].append(name)
            save_faces(data)

            return JsonResponse({"message": f"Face for {name} registered successfully!"})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    return render(request, "register.html")

def recognize_user(request):
    if request.method == "POST":
        image_data = request.POST.get("image")
        if not image_data:
            return JsonResponse({"message": "No image received!"})

        try:
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({"message": "Invalid image data."})

            # Resize if too large
            h, w = frame.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Detect faces
            face_locations, face_regions = detect_faces_mediapipe(frame)
            
            if not face_locations:
                return JsonResponse({"message": "No face detected."})

            data = load_faces()
            known_encs = data["encodings"]
            known_names = data["names"]

            results_out = []

            for location, face_roi in zip(face_locations, face_regions):
                # Face recognition
                name = "Unknown"
                encoding = get_face_encoding(face_roi)
                
                if encoding is not None and known_encs:
                    dists = face_recognition.face_distance(known_encs, encoding)
                    best_i = int(np.argmin(dists))
                    best_d = float(dists[best_i])
                    if best_d < 0.55:
                        name = known_names[best_i]

                # Emotion detection - use the improved method
                emotion, confidence = detect_emotion_with_opencv(face_roi)
                
                results_out.append(f"{name} ({emotion})")

            if results_out:
                return JsonResponse({"message": "Recognized: " + ", ".join(results_out)})
            else:
                return JsonResponse({"message": "No recognizable faces detected."})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    return render(request, "recognize.html")