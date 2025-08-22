from django.shortcuts import render
from django.http import JsonResponse
import cv2
import face_recognition
import pickle
import os
import base64
import numpy as np

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

def home(request):
    return render(request, "home.html")

def register_user(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image_data = request.POST.get("image")

        if not name or not image_data:
            return JsonResponse({"message": "Name and image are required!"})

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

    return render(request, "register.html")

def recognize_user(request):
    if request.method == "POST":
        image_data = request.POST.get("image")

        if not image_data:
            return JsonResponse({"message": "No image received!"})

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

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = data["names"][match_index]

            names.append(name)

        if names:
            return JsonResponse({"message": "Recognized: " + ", ".join(names)})
        else:
            return JsonResponse({"message": "No face detected."})

    return render(request, "recognize.html")
