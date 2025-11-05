# TODO: Integrate Emotion Detection Model

## Approved Plan
Replace EAR and MAR thresholding with the emotion detection model in face_project/model/emotion_model.h5.

## Steps
- [x] Update face_utils.py: Add TensorFlow/Keras imports, load the emotion model globally, and create detect_emotion_model function (expects 48x48 RGB input, normalized [0,1], outputs probabilities for 7 emotions).
- [x] Update views.py: Replace detect_emotion_with_opencv with detect_emotion_model from face_utils in process_frame and recognize_user functions.
- [x] Test the integration: Run the Django app and verify emotion detection uses the model.
- [x] Ensure dependencies: Confirm TensorFlow/Keras is installed; if not, install via pip.
