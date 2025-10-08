# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
import cv2
import mediapipe as mp
import pickle
import os
import base64
import numpy as np
import face_recognition
import time
from .models import EmotionRecord  # make sure this model exists

# ------------------------------------------------------------------
# Config and Initialization
# ------------------------------------------------------------------
FACE_DATA_FILE = "faces.pkl"

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_CONFIDENCE = 0.4

MIN_FACE_SIZE = 60
TARGET_FACE_SIZE = 160

# ------------------------------------------------------------------
# Emotion detection helpers (unchanged)
# ------------------------------------------------------------------
def detect_emotion_from_landmarks(face_img):
    try:
        if face_img.size == 0: return "Neutral", 0.5
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)
        if not results.multi_face_landmarks: return "Neutral", 0.5

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = face_img.shape[:2]
        try:
            lip_top, lip_bottom = landmarks[13], landmarks[14]
            mouth_left, mouth_right = landmarks[61], landmarks[291]
            left_eye_top, left_eye_bottom = landmarks[159], landmarks[145]
            right_eye_top, right_eye_bottom = landmarks[386], landmarks[374]

            mouth_height = abs(lip_bottom.y * h - lip_top.y * h)
            mouth_width = abs(mouth_right.x * w - mouth_left.x * w)
            mouth_aspect_ratio = mouth_height / (mouth_width + 1e-5)

            left_eye_openness = abs(left_eye_bottom.y * h - left_eye_top.y * h)
            right_eye_openness = abs(right_eye_bottom.y * h - right_eye_top.y * h)
            avg_eye_openness = (left_eye_openness + right_eye_openness) / 2

            emotions, confidences = [], []

            if mouth_aspect_ratio > 0.15:
                emotions.append("Happy")
                confidences.append(min(0.9, mouth_aspect_ratio * 3))
            if avg_eye_openness > 15:
                emotions.append("Surprise")
                confidences.append(min(0.9, avg_eye_openness / 25))
            if mouth_aspect_ratio < 0.01:
                emotions.append("Sad")
                confidences.append(0.6)

            if not emotions: return "Neutral", 0.6

            best_idx = np.argmax(confidences)
            best_confidence = confidences[best_idx]

            return (emotions[best_idx], best_confidence) if best_confidence >= EMOTION_CONFIDENCE else ("Neutral", best_confidence)
        except IndexError:
            return "Neutral", 0.5
    except Exception as e:
        print(f"Landmark emotion detection error: {e}")
        return "Neutral", 0.5


def detect_emotion_with_opencv(face_img):
    try:
        if face_img.size == 0: return "Neutral", 0.5
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if smile_cascade.empty():
            return detect_emotion_from_landmarks(face_img)
        smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:
            return "Happy", 0.7
        return detect_emotion_from_landmarks(face_img)
    except Exception as e:
        print(f"OpenCV emotion detection error: {e}")
        return "Neutral", 0.5

# ------------------------------------------------------------------
# Face helpers
# ------------------------------------------------------------------
def detect_faces_mediapipe(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    face_locations, face_regions = [], []
    if results.detections:
        for detection in results.detections:
            if detection.score[0] < 0.6: continue
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            width, height = min(width, w - x), min(height, h - y)
            if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                padding = int(min(width, height) * 0.15)
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(w, x + width + padding), min(h, y + height + padding)
                face_locations.append((y1, x2, y2, x1))
                face_region = frame[y1:y2, x1:x2]
                if face_region.size > 0:
                    face_regions.append(face_region)
    return face_locations, face_regions


def enhance_face_image(face_img):
    try:
        if face_img.size == 0: return None
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
    try:
        enhanced_face = enhance_face_image(face_img)
        if enhanced_face is None: return None
        encodings = face_recognition.face_encodings(enhanced_face)
        if encodings: return encodings[0]
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Face encoding error: {e}")
        return None

# ------------------------------------------------------------------
# Persistence: load/save faces
# ------------------------------------------------------------------
def load_faces():
    if os.path.exists(FACE_DATA_FILE):
        try:
            with open(FACE_DATA_FILE, "rb") as f:
                data = pickle.load(f)
                return data if isinstance(data, dict) and "encodings" in data else {"encodings": [], "names": []}
        except Exception:
            return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}


def save_faces(data):
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

# ------------------------------------------------------------------
# Confidence score logic (unchanged)
# ------------------------------------------------------------------
POSITIVE_EMOTIONS = ['Happy', 'Neutral']
NEGATIVE_EMOTIONS = ['Sad', 'Angry', 'Fear', 'Disgust']
TIMER_DURATION = 30  # seconds

def update_confidence_score(emotion, current_score):
    if emotion in POSITIVE_EMOTIONS:
        return current_score + 1
    elif emotion in NEGATIVE_EMOTIONS:
        return current_score - 1
    return current_score

# ------------------------------------------------------------------
# Emotion timeframe / recording (unchanged function)
# ------------------------------------------------------------------
def update_emotion_timeframe(name, emotion):
    """
    Updates or creates a record of how many times a user displayed a specific emotion.
    Tracks first and last detection times.
    """
    if not name or name == "Unknown":
        return

    record, created = EmotionRecord.objects.get_or_create(
        name=name,
        emotion=emotion,
        defaults={"count": 1, "first_detected": timezone.now()}
    )

    if not created:
        record.count += 1
        record.last_detected = timezone.now()
        record.save()

def emotion_stats(request):
    data = EmotionRecord.objects.all().values("name", "emotion", "count", "first_detected", "last_detected")
    return JsonResponse(list(data), safe=False)

# ------------------------------------------------------------------
# New endpoints to control counting (start/stop and status)
# ------------------------------------------------------------------
def start_counting(request):
    if request.method == "POST":
        request.session['counting_active'] = True
        return JsonResponse({"status":"success", "counting": True, "message": "Counting started."})
    return JsonResponse({"status":"error", "message":"Invalid request method."}, status=405)

def stop_counting(request):
    if request.method == "POST":
        request.session['counting_active'] = False
        return JsonResponse({"status":"success", "counting": False, "message": "Counting stopped."})
    return JsonResponse({"status":"error", "message":"Invalid request method."}, status=405)

def counting_status(request):
    return JsonResponse({"counting": bool(request.session.get('counting_active', False))})

# ------------------------------------------------------------------
# Confidence Test Views (process_frame uses counting flag)
# ------------------------------------------------------------------
def start_confidence_test(request):
    if request.method == "POST":
        request.session['test_start_time'] = time.time()
        request.session['confidence_score'] = 50
        return JsonResponse({"message": "Test started.", "status": "success"})
    return JsonResponse({"message": "Invalid request method.", "status": "error"}, status=405)


def process_frame(request):
    if request.method != "POST":
        return JsonResponse({"message": "Invalid request method.", "status": "error"}, status=405)

    if 'test_start_time' not in request.session:
        return JsonResponse({"message": "Test not started.", "error": "not_started"})

    img_data = request.POST.get("image")
    if not img_data:
        return JsonResponse({"message": "No image received!", "status": "error"})

    try:
        image_bytes = base64.b64decode(img_data.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return JsonResponse({"message": "Invalid image format.", "status": "error"})

    face_locations, face_regions = detect_faces_mediapipe(frame)
    current_score = request.session.get('confidence_score', 50)
    name, emotion = "Unknown", "Neutral"

    if face_regions:
        face_roi = face_regions[0]
        encoding = get_face_encoding(face_roi)
        if encoding is not None:
            face_data = load_faces()
            if face_data["encodings"]:
                dists = face_recognition.face_distance(face_data["encodings"], encoding)
                best_i = np.argmin(dists)
                if dists[best_i] < 0.6:
                    name = face_data["names"][best_i]

        emotion, _ = detect_emotion_with_opencv(face_roi)

        new_score = update_confidence_score(emotion, current_score)
        request.session['confidence_score'] = new_score

        # ONLY record emotion counts WHEN counting is active in session
        if request.session.get('counting_active', False):
            update_emotion_timeframe(name, emotion)

    start_time = request.session['test_start_time']
    elapsed_time = time.time() - start_time

    if elapsed_time >= TIMER_DURATION:
        final_score = request.session['confidence_score']
        del request.session['test_start_time']
        del request.session['confidence_score']
        return JsonResponse({"final": True, "name": name, "emotion": emotion, "confidence": final_score})
    else:
        return JsonResponse({
            "final": False,
            "name": name,
            "emotion": emotion,
            "confidence": request.session['confidence_score'],
            "time_left": round(TIMER_DURATION - elapsed_time)
        })

# ------------------------------------------------------------------
# Existing register/recognize/home views (recognize_user also checks counting flag)
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
            face_locations, face_regions = detect_faces_mediapipe(frame)
            if not face_locations:
                return JsonResponse({"message": "No face detected. Try again with better lighting."})
            face_roi = face_regions[0]
            encoding = get_face_encoding(face_roi)
            if encoding is None:
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
            face_locations, face_regions = detect_faces_mediapipe(frame)
            if not face_locations:
                return JsonResponse({"message": "No face detected."})
            data = load_faces()
            known_encs, known_names = data["encodings"], data["names"]
            results_out = []
            for location, face_roi in zip(face_locations, face_regions):
                name = "Unknown"
                encoding = get_face_encoding(face_roi)
                if encoding is not None and known_encs:
                    dists = face_recognition.face_distance(known_encs, encoding)
                    best_i = int(np.argmin(dists))
                    if dists[best_i] < 0.6:
                        name = known_names[best_i]
                emotion, confidence = detect_emotion_with_opencv(face_roi)
                results_out.append(f"{name} ({emotion})")

                # ONLY record emotion counts WHEN counting is active in session
                if request.session.get('counting_active', False):
                    update_emotion_timeframe(name, emotion)

            return JsonResponse({"message": "Recognized: " + ", ".join(results_out)})
        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})
    return render(request, "recognize.html")
