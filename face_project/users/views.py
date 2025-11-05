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
from .models import EmotionRecord  # ensure this model exists
from django.views.decorators.csrf import csrf_exempt
from .face_utils import detect_emotion_model

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
# Emotion detection helpers
# ------------------------------------------------------------------
def detect_emotion_from_landmarks(face_img):
    try:
        if face_img is None or face_img.size == 0:
            return "Neutral", 0.5

        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)
        if not results.multi_face_landmarks:
            return "Neutral", 0.5

        lm = results.multi_face_landmarks[0].landmark
        h, w = face_img.shape[:2]

        def P(i): return np.array([lm[i].x * w, lm[i].y * h])

        LE, RE = [33,160,158,133,153,144], [263,387,385,362,380,373]
        M_TOP, M_BOTTOM, M_L, M_R = 13, 14, 61, 291
        BROW_L, BROW_R, EYE_TOP_L, EYE_TOP_R = 70, 300, 159, 386

        def eye_ear(indices):
            p = [P(i) for i in indices]
            vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
            horz = np.linalg.norm(p[0] - p[3]) + 1e-6
            return vert / (2.0 * horz)

        ear = (eye_ear(LE) + eye_ear(RE)) / 2.0
        mar = np.linalg.norm(P(M_BOTTOM) - P(M_TOP)) / (np.linalg.norm(P(M_R) - P(M_L)) + 1e-6)
        mouth_width_norm = np.linalg.norm(P(M_R) - P(M_L)) / float(w)
        brow_eye_norm = ((P(EYE_TOP_L)[1] - P(BROW_L)[1]) + (P(EYE_TOP_R)[1] - P(BROW_R)[1])) / (2.0 * h)
        lip_center = (P(M_TOP) + P(M_BOTTOM)) / 2.0
        corner_up_norm = ((lip_center[1] - P(M_L)[1]) + (lip_center[1] - P(M_R)[1])) / (2.0 * h)

        emotion, conf = "Neutral", 0.5

        if mar > 0.35 and ear > 0.28 and brow_eye_norm > 0.03:
            emotion, conf = "Surprise", 0.95
        elif ear > 0.07 and mar > 0.2:
            emotion, conf = "Happy", 0.9
        elif brow_eye_norm < 0.01 and ear < 0.16 and mar < 0.18:
            emotion, conf = "Angry", 0.88
        elif brow_eye_norm > 0.05 and ear > 0.50 and mar < 0.25:
            emotion, conf = "Fear", 0.78
        elif ear < 0.2 and mar < 0.1:
            emotion, conf = "Sad", 0.85
        elif mouth_width_norm < 0.18 and corner_up_norm > 0.005 and mar < 0.12:
            emotion, conf = "Disgust", 0.70
        else:
            emotion, conf = "Neutral", 0.6

        if conf < EMOTION_CONFIDENCE:
            return "Neutral", conf
        return emotion, conf

    except Exception as e:
        print("Landmark emotion detection error:", e)
        return "Neutral", 0.5


def detect_emotion_with_opencv(face_img):
    try:
        if face_img is None or face_img.size == 0:
            return "Neutral", 0.5

        emotion, conf = detect_emotion_from_landmarks(face_img)
        if emotion != "Neutral" or conf >= EMOTION_CONFIDENCE:
            return emotion, conf

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        if not smile_cascade.empty():
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20, minSize=(25,25))
            if len(smiles) > 0:
                return "Happy", 0.8

        return "Neutral", 0.5
    except Exception as e:
        print("OpenCV emotion detection error:", e)
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
            x, y = int(bbox.xmin*w), int(bbox.ymin*h)
            width, height = int(bbox.width*w), int(bbox.height*h)
            x, y = max(0, x), max(0, y)
            width, height = min(width, w-x), min(height, h-y)
            if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                pad = int(min(width, height)*0.15)
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(w, x+width+pad), min(h, y+height+pad)
                face_locations.append((y1,x2,y2,x1))
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_regions.append(face)
    return face_locations, face_regions


def enhance_face_image(face_img):
    if face_img.size == 0: return None
    h, w = face_img.shape[:2]
    if min(h, w) < TARGET_FACE_SIZE:
        scale = TARGET_FACE_SIZE / float(min(h, w))
        face_img = cv2.resize(face_img, (int(w*scale), int(h*scale)))
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    ycrcb = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def get_face_encoding(face_img):
    try:
        enhanced = enhance_face_image(face_img)
        if enhanced is None: return None
        encs = face_recognition.face_encodings(enhanced)
        if encs: return encs[0]
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        return encs[0] if encs else None
    except Exception as e:
        print("Face encoding error:", e)
        return None

# ------------------------------------------------------------------
# Persistence: load/save faces
# ------------------------------------------------------------------
def load_faces():
    if os.path.exists(FACE_DATA_FILE):
        try:
            with open(FACE_DATA_FILE, "rb") as f:
                data = pickle.load(f)
                return data if "encodings" in data else {"encodings": [], "names": []}
        except Exception:
            return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}


def save_faces(data):
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

# ------------------------------------------------------------------
# Confidence scoring
# ------------------------------------------------------------------
POSITIVE_EMOTIONS = ['Happy', 'Neutral']
NEGATIVE_EMOTIONS = ['Sad', 'Angry', 'Fear', 'Disgust']
TIMER_DURATION = 30

def update_confidence_score(emotion, score):
    if emotion in POSITIVE_EMOTIONS:
        return score + 2
    elif emotion in NEGATIVE_EMOTIONS:
        return score - 2
    return score

# ------------------------------------------------------------------
# Emotion timeframe tracking (DB)
# ------------------------------------------------------------------
def update_emotion_timeframe(name, emotion):
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
# Counting controls (start/stop/status)
# ------------------------------------------------------------------
def start_counting(request):
    if request.method == "POST":
        request.session['counting_active'] = True
        request.session['confidence_score'] = 0
        return JsonResponse({"status":"success", "counting": True, "message":"Counting started."})
    return JsonResponse({"status":"error", "message":"Invalid request."}, status=405)

def stop_counting(request):
    if request.method == "POST":
        final_conf = request.session.get('confidence_score', 0)
        request.session['counting_active'] = False
        return JsonResponse({
            "status":"success",
            "counting": False,
            "message":"Counting stopped.",
            "final_confidence": final_conf
        })
    return JsonResponse({"status":"error", "message":"Invalid request."}, status=405)

def counting_status(request):
    return JsonResponse({"counting": bool(request.session.get('counting_active', False))})

# ------------------------------------------------------------------
# Confidence test and processing frames
# ------------------------------------------------------------------
def start_confidence_test(request):
    if request.method == "POST":
        request.session['test_start_time'] = time.time()
        request.session['confidence_score'] = 0
        return JsonResponse({"message":"Test started.","status":"success"})
    return JsonResponse({"message":"Invalid request.","status":"error"}, status=405)

def process_frame(request):
    if request.method != "POST":
        return JsonResponse({"message":"Invalid request.","status":"error"}, status=405)

    if 'test_start_time' not in request.session:
        return JsonResponse({"message":"Test not started.","error":"not_started"})

    img_data = request.POST.get("image")
    if not img_data:
        return JsonResponse({"message":"No image received!","status":"error"})

    try:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_data.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return JsonResponse({"message":"Invalid image format.","status":"error"})

    face_locations, face_regions = detect_faces_mediapipe(frame)
    score = request.session.get('confidence_score', 0)
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

        emotion, _ = detect_emotion_model(face_roi)
        score = update_confidence_score(emotion, score)
        request.session['confidence_score'] = score

        if request.session.get('counting_active', False):
            update_emotion_timeframe(name, emotion)

    elapsed = time.time() - request.session['test_start_time']
    if elapsed >= TIMER_DURATION:
        final = request.session['confidence_score']
        del request.session['test_start_time']
        del request.session['confidence_score']
        return JsonResponse({"final":True, "name":name, "emotion":emotion, "confidence":final})
    else:
        return JsonResponse({
            "final":False,
            "name":name,
            "emotion":emotion,
            "confidence":score,
            "time_left": round(TIMER_DURATION - elapsed)
        })

# ------------------------------------------------------------------
# Home, Register, Recognize views
# ------------------------------------------------------------------
def home(request):
    return render(request, "home.html")

def register_user(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        img = request.POST.get("image")
        if not name or not img:
            return JsonResponse({"message":"Name and image required!"})
        try:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            face_locations, face_regions = detect_faces_mediapipe(frame)
            if not face_regions:
                return JsonResponse({"message":"No face detected."})
            face_roi = face_regions[0]
            encoding = get_face_encoding(face_roi)
            if encoding is None:
                return JsonResponse({"message":"Could not extract features."})
            data = load_faces()
            if data["encodings"]:
                dists = face_recognition.face_distance(data["encodings"], encoding)
                best_i = np.argmin(dists)
                if dists[best_i] < 0.4 and data["names"][best_i].lower() == name.lower():
                    return JsonResponse({"message":f"{name} already registered."})
            data["encodings"].append(encoding)
            data["names"].append(name)
            save_faces(data)
            return JsonResponse({"message":f"Face for {name} registered successfully!"})
        except Exception as e:
            return JsonResponse({"message":f"Error: {e}"})
    return render(request, "register.html")

def recognize_user(request):
    if request.method == "POST":
        img = request.POST.get("image")
        if not img:
            return JsonResponse({"message":"No image received!"})
        try:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            face_locations, face_regions = detect_faces_mediapipe(frame)
            if not face_regions:
                return JsonResponse({"message":"No face detected."})
            data = load_faces()
            results_out = []
            for face_roi in face_regions:
                name = "Unknown"
                enc = get_face_encoding(face_roi)
                if enc is not None and data["encodings"]:
                    dists = face_recognition.face_distance(data["encodings"], enc)
                    best_i = np.argmin(dists)
                    if dists[best_i] < 0.6:
                        name = data["names"][best_i]
                emotion, _ = detect_emotion_model(face_roi)
                results_out.append(f"{name} ({emotion})")

                if request.session.get('counting_active', False):
                    update_emotion_timeframe(name, emotion)

            return JsonResponse({"message":"Recognized: " + ", ".join(results_out)})
        except Exception as e:
            return JsonResponse({"message":f"Error: {e}"})
    return render(request, "recognize.html")

# ------------------------------------------------------------------
# Delete all emotion records
# ------------------------------------------------------------------
@csrf_exempt
def delete_emotions(request):
    if request.method == "POST":
        EmotionRecord.objects.all().delete()
        return JsonResponse({"message": "All emotion records deleted successfully."})
    return JsonResponse({"error": "Invalid request method."}, status=405)
