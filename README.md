
# 🎭 Real-Time Face & Emotion Recognition with Confidence Scoring

A real-time web-based application that performs **face recognition**, **emotion detection**, and **confidence score computation** using a live camera feed.
The system tracks emotions per user and computes a confidence score based on emotional patterns during the session.

---

## 🚀 Features

* 📸 Live webcam face recognition
* 🙂 Emotion detection (Happy, Neutral, Sad, Angry, etc.)
* 🔢 Emotion count tracking per user
* 🧠 Confidence score calculation based on emotions
* ▶️ Start / Stop emotion counting
* 📊 Real-time emotion statistics table
* 🗑️ Delete all emotion records
* 🔐 Secure backend with CSRF protection

---

## 🧠 Confidence Score Logic

The confidence score is computed dynamically using emotion counts:

| Emotion          | Score Impact |
| ---------------- | ------------ |
| Happy            | +2           |
| Neutral          | +2           |
| Sad              | -1           |
| Angry            | -2           |
| Disgust / Others | -2           |

The **final confidence score** is displayed when the counting process stops.

---

## 🛠️ Tech Stack

**Frontend**

* HTML5
* CSS3 (Modern UI / Glassmorphism)
* JavaScript (Fetch API)

**Backend**

* Python (Django)
* OpenCV
* `face_recognition` library
* Deep Learning emotion classifier

**Storage**

* SQLite database
* `faces.pkl` for face encodings

---

## 📂 Project Structure

```
FACE_REC/
│
├── face_project/              # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── model/                     # ML / DL emotion recognition models
│   ├── emotion_model.h5
│   └── model_utils.py
│
├── templates/                 # HTML templates
│   └── index.html
│
├── users/                     # Django app (core logic)
│   ├── migrations/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── utils.py
│
├── db.sqlite3                 # SQLite database
├── faces.pkl                  # Stored face encodings
├── manage.py                  # Django entry point
└── README.md
```

---

## ⚙️ How It Works

1. Webcam captures live video frames.
2. Faces are detected and encoded.
3. Face encodings are matched against `faces.pkl`.
4. Emotions are predicted using a trained DL model.
5. Emotion counts are stored in the database.
6. Confidence score is calculated from emotion statistics.
7. Results are displayed live on the UI.

---

## ▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/face-recognition-emotion.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run migrations:

```bash
python manage.py migrate
```

4. Start the server:

```bash
python manage.py runserver
```

5. Open in browser:

```
http://127.0.0.1:8000/
```

---

## 🔐 Data & Privacy

* Camera access requires explicit user permission
* Face encodings stored locally in `faces.pkl`
* Emotion records can be deleted from UI
* No cloud storage by default

---

## 📌 Real-World Use Cases

* Online learning engagement analysis
* Interview confidence evaluation
* Mental health monitoring
* Smart classrooms
* Corporate training analytics
* Customer experience research

---

## 🔮 Future Enhancements

* Multi-face tracking
* Emotion trend graphs
* Audio sentiment analysis
* Role-based user access
* Cloud deployment
* Advanced transformer-based emotion models

---

## 👩‍💻 Author

**Punnavajhala Nagasrisreya**
Projects focused on **Machine Learning, Deep Learning & Computer Vision**

