from django.shortcuts import render
from django.http import JsonResponse
import cv2
import face_recognition
import pickle
import os
import base64
import numpy as np
from fer import FER  # Facial Expression Recognition

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FACE_DATA_FILE = "faces.pkl"

# Face match threshold (smaller = stricter)
TOLERANCE = 0.46

# Minimum face size (in original frame pixels) to attempt FER
MIN_FACE_H = 60
MIN_FACE_W = 60

# Downscale factor for faster detection/encoding
DOWNSCALE = 0.5  # 50% size

# Confidence threshold for accepting FER's top emotion
EMOTION_CONFIDENCE = 0.45

# Initialize FER once (mtcnn=False avoids heavy torch/mtcnn stack)
expression_detector = FER(mtcnn=False)


# ------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------
def load_faces():
    """Load saved encodings & names (safe default if missing)."""
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, "rb") as f:
            data = pickle.load(f)
            # Basic sanity check
            if not isinstance(data, dict):
                return {"encodings": [], "names": []}
            return {
                "encodings": list(data.get("encodings", [])),
                "names": list(data.get("names", [])),
            }
    return {"encodings": [], "names": []}


def save_faces(data):
    """Save encodings & names."""
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)


# ------------------------------------------------------------------
# Views
# ------------------------------------------------------------------
def home(request):
    return render(request, "home.html")


def register_user(request):
    """Register a new user with one snapshot."""
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        image_data = request.POST.get("image")

        if not name or not image_data:
            return JsonResponse({"message": "Name and image are required!"})

        try:
            # Decode base64 image -> BGR frame
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({"message": "Invalid image data."})

            # Work in RGB for face_recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Downscale for speed
            small_rgb = cv2.resize(rgb, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            face_locations_small = face_recognition.face_locations(
                small_rgb, number_of_times_to_upsample=1, model="hog"
            )
            face_encodings_small = face_recognition.face_encodings(
                small_rgb, face_locations_small
            )

            if not face_encodings_small:
                return JsonResponse({"message": "No face detected. Try again."})

            # Use the first detected face
            encoding = face_encodings_small[0]

            # Prevent duplicates: if this face is already very close to an existing encoding,
            # don't add a near-identical copy (esp. for same name).
            data = load_faces()
            if data["encodings"]:
                dists = face_recognition.face_distance(data["encodings"], encoding)
                best_i = int(np.argmin(dists))
                best_d = float(dists[best_i])
                if best_d < 0.35 and data["names"][best_i].lower() == name.lower():
                    return JsonResponse({"message": f"{name} already registered (similar face found)."})
            # Save
            data["encodings"].append(encoding)
            data["names"].append(name)
            save_faces(data)

            return JsonResponse({"message": f"Face for {name} registered successfully!"})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    # GET
    return render(request, "register.html")


def recognize_user(request):
    """Recognize faces and estimate expressions."""
    if request.method == "POST":
        image_data = request.POST.get("image")
        if not image_data:
            return JsonResponse({"message": "No image received!"})

        try:
            # Decode base64 image -> BGR frame
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({"message": "Invalid image data."})

            # Prepare RGB for face_recognition
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Downscale for speed
            small_rgb = cv2.resize(rgb, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)

            # Detect & encode on the small image
            face_locations_small = face_recognition.face_locations(
                small_rgb, number_of_times_to_upsample=1, model="hog"
            )
            face_encodings_small = face_recognition.face_encodings(
                small_rgb, face_locations_small
            )

            # Map locations back to original frame (BGR for FER ROI)
            scale = 1.0 / DOWNSCALE
            face_locations = [
                (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))
                for (top, right, bottom, left) in face_locations_small
            ]

            data = load_faces()
            known_encs = data["encodings"]
            known_names = data["names"]

            results_out = []

            for enc, (top, right, bottom, left) in zip(face_encodings_small, face_locations):
                # ----- Name by distance + threshold -----
                name = "Unknown"
                if known_encs:
                    dists = face_recognition.face_distance(known_encs, enc)
                    best_i = int(np.argmin(dists))
                    best_d = float(dists[best_i])
                    if best_d < TOLERANCE:
                        name = known_names[best_i]

                # ----- Expression on sufficiently large ROI -----
                emotion = "Neutral"
                # Use BGR ROI for FER (FER expects BGR by default)
                face_bgr = frame[top:bottom, left:right]

                if face_bgr.size > 0:
                    h, w = face_bgr.shape[:2]
                    if h >= MIN_FACE_H and w >= MIN_FACE_W:
                        try:
                            # Resize for stability & speed
                            roi = cv2.resize(face_bgr, (224, 224))
                            fer_out = expression_detector.detect_emotions(roi)
                            if fer_out and "emotions" in fer_out[0]:
                                # Choose only if confidence is reasonable
                                emo_dict = fer_out[0]["emotions"]
                                best_emo, best_score = max(emo_dict.items(), key=lambda kv: kv[1])
                                emotion = best_emo if best_score >= EMOTION_CONFIDENCE else "Neutral"
                        except Exception:
                            # Keep emotion as "Neutral" if FER fails
                            pass

                results_out.append(f"{name} ({emotion})")

            if results_out:
                return JsonResponse({"message": "Recognized: " + ", ".join(results_out)})
            else:
                return JsonResponse({"message": "No face detected."})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    # GET -> show webcam page
    return render(request, "recognize.html")
