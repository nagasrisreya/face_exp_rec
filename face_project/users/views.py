from django.shortcuts import render
from django.http import JsonResponse
import cv2
import face_recognition
import pickle
import os
import base64
import numpy as np
from fer import FER  # Facial Expression Recognition

# File to store known faces
FACE_DATA_FILE = "faces.pkl"

# Load saved encodings if available
def load_faces():
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

# Save encodings
def save_faces(data):
    with open(FACE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)

# Home page
def home(request):
    return render(request, "home.html")

# Register user with webcam snapshot
def register_user(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image_data = request.POST.get("image")

        if not name or not image_data:
            return JsonResponse({"message": "Name and image are required!"})

        try:
            # Decode base64 image
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                data = load_faces()
                data["encodings"].append(face_encodings[0])
                data["names"].append(name)
                save_faces(data)
                return JsonResponse({"message": f"Face for {name} registered successfully!"})
            else:
                return JsonResponse({"message": "No face detected. Try again."})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    return render(request, "register.html")

# Recognize user + detect expressions
def recognize_user(request):
    if request.method == "POST":
        image_data = request.POST.get("image")

        if not image_data:
            return JsonResponse({"message": "No image received!"})

        try:
            # Decode base64 image
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            data = load_faces()
            names = []
            expressions = []

            # Initialize FER detector
            detector = FER(mtcnn=True)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(data["encodings"], face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = data["names"][match_index]

                names.append(name)

                # Detect facial expression for this face
                top, right, bottom, left = face_location
                face_roi = rgb_frame[top:bottom, left:right]

                if face_roi.size > 0:  # Ensure region isn't empty
                    result = detector.detect_emotions(face_roi)
                    if result:
                        # Get dominant emotion
                        emotion, score = max(result[0]["emotions"].items(), key=lambda x: x[1])
                        expressions.append(emotion)
                    else:
                        expressions.append("Neutral")
                else:
                    expressions.append("Neutral")

            if names:
                results = [f"{n} ({e})" for n, e in zip(names, expressions)]
                return JsonResponse({"message": "Recognized: " + ", ".join(results)})
            else:
                return JsonResponse({"message": "No face detected."})

        except Exception as e:
            return JsonResponse({"message": f"Error: {str(e)}"})

    # When GET request, show webcam recognize page
    return render(request, "recognize.html")
