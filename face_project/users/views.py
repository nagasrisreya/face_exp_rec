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
import numpy as np

def detect_emotion_from_landmarks(face_img):
    """
    Improved rule-based emotion detector using FaceMesh landmarks.
    Uses EAR (eye aspect ratio), MAR (mouth aspect ratio), brow-eye distance,
    and mouth-corner geometry to infer emotions.
    Returns: (emotion_str, confidence_float)
    """
    try:
        if face_img is None or face_img.size == 0:
            return "Neutral", 0.5

        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)
        if not results.multi_face_landmarks:
            return "Neutral", 0.5

        lm = results.multi_face_landmarks[0].landmark
        h, w = face_img.shape[:2]

        # helper to get (x,y) in pixel coords
        def P(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        # Mediapipe face mesh indices (commonly used sets)
        LE = [33, 160, 158, 133, 153, 144]     # left eye
        RE = [263, 387, 385, 362, 380, 373]    # right eye
        # mouth top/bottom and corners
        M_TOP, M_BOTTOM = 13, 14
        M_L, M_R = 61, 291
        # eyebrow / eye reference points
        BROW_L, BROW_R = 70, 300
        EYE_TOP_L, EYE_TOP_R = 159, 386

        # compute EAR for an eye (based on 6 points: p1..p6)
        def eye_ear(indices):
            p = [P(i) for i in indices]
            # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
            horz = np.linalg.norm(p[0] - p[3]) + 1e-6
            return vert / (2.0 * horz)

        ear_l = eye_ear(LE)
        ear_r = eye_ear(RE)
        ear = (ear_l + ear_r) / 2.0

        # mouth aspect ratio (MAR)
        top = P(M_TOP); bottom = P(M_BOTTOM)
        left_m = P(M_L); right_m = P(M_R)
        mouth_vert = np.linalg.norm(bottom - top)
        mouth_horz = np.linalg.norm(right_m - left_m) + 1e-6
        mar = mouth_vert / mouth_horz

        # mouth width normalized
        mouth_width_norm = mouth_horz / float(w)

        # eyebrow distance relative to eyes (brow raise / lower)
        brow_l = P(BROW_L); brow_r = P(BROW_R)
        eye_top_l = P(EYE_TOP_L); eye_top_r = P(EYE_TOP_R)
        brow_eye_dist = ((eye_top_l[1] - brow_l[1]) + (eye_top_r[1] - brow_r[1])) / 2.0
        brow_eye_norm = brow_eye_dist / float(h)   # bigger => brows raised

        # mouth corner relative to lip center (smile indicator)
        lip_center = (top + bottom) / 2.0
        corner_up = ((lip_center[1] - left_m[1]) + (lip_center[1] - right_m[1])) / 2.0
        corner_up_norm = corner_up / float(h)   # positive => corners higher than lip center (smile)

        # Small normalization transforms to make thresholds more stable:
        # (these scale values into comfortable ranges for thresholds below)
        # ear typical range ~ 0.10 - 0.35, mar typical ~ 0.02 - 0.6
        # Decide emotion by rules and assign confidences
        emotion = "Neutral"
        conf = 0.5

        # ----- Rules (ordered by more-specific -> less-specific) -----
        # Surprise: very open mouth, wide eyes, brows raised
        if mar > 0.35 and ear > 0.28 and brow_eye_norm > 0.03:
            emotion, conf = "Surprise", 0.95

        # Big smile: mouth open enough and corners up
        elif corner_up_norm > 0.015 and mar > 0.22:
            emotion, conf = "Happy", 0.9

        # Angry: brows lowered (small brow_eye_norm), narrow eyes, mouth tight
        elif brow_eye_norm < 0.01 and ear < 0.16 and mar < 0.18:
            emotion, conf = "Angry", 0.88

        # Fear: brows somewhat raised but mouth not as wide as surprise, eyes open
        elif brow_eye_norm > 0.05 and ear > 0.50 and mar < 0.25:
            emotion, conf = "Fear", 0.78

        # Sad: eyes droopy (low EAR) and mouth small / corners down
        elif ear < 0.2 and mar < 0.2:
            emotion, conf = "Sad", 0.85

        # Disgust (approx): mouth narrow and upper lip raised (corners slightly up but small width)
        elif mouth_width_norm < 0.18 and corner_up_norm > 0.005 and mar < 0.12:
            emotion, conf = "Disgust", 0.70

        # Neutral fallback
        else:
            emotion, conf = "Neutral", 0.6

        # Respect your EMOTION_CONFIDENCE threshold
        if conf < EMOTION_CONFIDENCE:
            return "Neutral", conf
        return emotion, conf

    except Exception as e:
        print(f"Landmark emotion detection error: {e}")
        return "Neutral", 0.5


def detect_emotion_with_opencv(face_img):
    """
    Try landmark-based detector first (better). Fallback to smile cascade.
    """
    try:
        if face_img is None or face_img.size == 0:
            return "Neutral", 0.5

        # Landmark-based detection (preferred)
        emotion, conf = detect_emotion_from_landmarks(face_img)
        if emotion != "Neutral" or conf >= EMOTION_CONFIDENCE:
            return emotion, conf

        # Fallback: Haar smile detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if not smile_cascade.empty():
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            if len(smiles) > 0:
                return "Happy", 0.8

        return "Neutral", 0.5

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
