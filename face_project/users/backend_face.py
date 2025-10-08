import cv2
import numpy as np
import base64
from . import face_utils

def recognize_face(img_data):
    try:
        image_bytes = base64.b64decode(img_data.split(",")[1])
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None, "Invalid image"

    if frame is None:
        return None, "Invalid frame"

    faces = face_utils.detect_faces(frame)
    if not faces:
        return None, "No face detected"

    data = face_utils.load_faces()
    known_encodings = [np.array(enc) for enc in data.get("encodings", [])]
    known_names = data.get("names", [])

    name = "Unknown"
    enc = face_utils.get_encoding(faces[0])

    if enc is not None and len(known_encodings) > 0:
        dists = face_utils.face_recognition.face_distance(known_encodings, enc)
        best_i = int(np.argmin(dists))
        if dists[best_i] < 0.5:
            name = known_names[best_i]

    return name, None
