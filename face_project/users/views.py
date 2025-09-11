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

# Minimum face size for processing (lowered for better distance tolerance)
MIN_FACE_SIZE = 60  
TARGET_FACE_SIZE = 160  # for upscaling small faces

# ------------------------------------------------------------------
# Enhanced Emotion Detection using Facial Landmarks
# ------------------------------------------------------------------
def detect_emotion_from_landmarks(face_img):
    """Detect emotion using facial landmarks and geometric features"""
    try:
        if face_img.size == 0:
            return "Neutral", 0.5
            
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)
        
        if not results.multi_face_landmarks:
            return "Neutral", 0.5
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = face_img.shape[:2]
        
        try:
            lip_top = landmarks[13]
            lip_bottom = landmarks[14]
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            
            mouth_height = abs(lip_bottom.y * h - lip_top.y * h)
            mouth_width = abs(mouth_right.x * w - mouth_left.x * w)
            mouth_aspect_ratio = mouth_height / (mouth_width + 1e-5)
            
            left_eye_openness = abs(left_eye_bottom.y * h - left_eye_top.y * h)
            right_eye_openness = abs(right_eye_bottom.y * h - right_eye_top.y * h)
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
            
            emotions = []
            confidences = []
            
            if mouth_aspect_ratio > 0.15:
                emotions.append("Happy")
                confidences.append(min(0.9, mouth_aspect_ratio * 3))
            
            if avg_eye_openness > 15:
                emotions.append("Surprise")
                confidences.append(min(0.9, avg_eye_openness / 25))
            
            if mouth_aspect_ratio < 0.01:
                emotions.append("Sad")
                confidences.append(0.6)
            
            if not emotions:
                return "Neutral", 0.6
                
            best_idx = np.argmax(confidences)
            best_confidence = confidences[best_idx]
            
            if best_confidence >= EMOTION_CONFIDENCE:
                return emotions[best_idx], best_confidence
            else:
                return "Neutral", best_confidence
                
        except IndexError:
            return "Neutral", 0.5
            
    except Exception as e:
        print(f"Landmark emotion detection error: {e}")
        return "Neutral", 0.5

def detect_emotion_with_opencv(face_img):
    """Alternative emotion detection using OpenCV features"""
    try:
        if face_img.size == 0:
            return "Neutral", 0.5
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if smile_cascade.empty():
            return detect_emotion_from_landmarks(face_img)
        
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        if len(smiles) > 0:
            return "Happy", 0.7
        
        return detect_emotion_from_landmarks(face_img)
        
    except Exception as e:
        print(f"OpenCV emotion detection error: {e}")
        return "Neutral", 0.5

# ------------------------------------------------------------------
# Enhanced Face Processing Functions
# ------------------------------------------------------------------
def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe with better range tolerance"""
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
            
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
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
    """Enhance + resize small faces for better recognition"""
    try:
        if face_img.size == 0:
            return None
        
        h, w = face_img.shape[:2]
        if min(h, w) < TARGET_FACE_SIZE:
            scale = TARGET_FACE_SIZE / float(min(h, w))
            face_img = cv2.resize(face_img, (int(w * scale), int(h * scale)))
        
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        if len(rgb_face.shape) == 3:
            ycrcb = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            rgb_face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        return rgb_face
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return None

def get_face_encoding(face_img):
    """Extract face encoding with better range handling"""
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

            h, w = frame.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

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

            data = load_faces()
            if data["encodings"]:
                dists = face_recognition.face_distance(data["encodings"], encoding)
                best_i = int(np.argmin(dists))
                if dists[best_i] < 0.4 and data["names"][best_i].lower() == name.lower():
                    return JsonResponse({"message": f"{name} is already registered."})
            
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

            h, w = frame.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            face_locations, face_regions = detect_faces_mediapipe(frame)
            
            if not face_locations:
                return JsonResponse({"message": "No face detected."})

            data = load_faces()
            known_encs = data["encodings"]
            known_names = data["names"]

            results_out = []

            for location, face_roi in zip(face_locations, face_regions):
                name = "Unknown"
                encoding = get_face_encoding(face_roi)
                
                if encoding is not None and known_encs:
                    dists = face_recognition.face_distance(known_encs, encoding)
                    best_i = int(np.argmin(dists))
                    if dists[best_i] < 0.6:  # slightly looser for range tolerance
                        name = known_names[best_i]

                emotion, confidence = detect_emotion_with_opencv(face_roi)
                
                results_out.append(f"{name} ({emotion})")

            if results_out:
                return JsonResponse({"message": "Recognized: " + ", ".join(results_out)})
            else:
                return JsonResponse({"message": "No recognizable faces detected."})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    return render(request, "recognize.html")
