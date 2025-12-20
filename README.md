
# 🎭 Real-Time Face & Emotion Recognition with Confidence Scoring

A real-time web-based system that performs **face recognition**, **emotion detection**, and **confidence score analysis** using a live camera feed.
The project dynamically tracks emotions per user and computes a confidence score based on emotional trends.

---

## 🚀 Features

* 📸 **Live Webcam Face Recognition**
* 🙂 **Emotion Detection** (Happy, Neutral, Sad, Angry, etc.)
* 🔢 **Emotion Count Tracking**
* 🧠 **Dynamic Confidence Score Calculation**

  * Positive emotions increase confidence
  * Negative emotions reduce confidence
* ▶️ **Start / Stop Emotion Counting**
* 📊 **Real-time Emotion Statistics Table**
* 🗑️ **Delete All Emotion Records**
* 🔐 **CSRF-protected backend communication**

---

## 🧠 Confidence Score Logic

Confidence is calculated **on the frontend** using emotion counts:

| Emotion          | Weight |
| ---------------- | ------ |
| Happy            | +2     |
| Neutral          | +2     |
| Sad              | -1     |
| Angry            | -2     |
| Disgust / Others | -2     |

The final confidence score updates automatically based on table data when counting stops.

---

## 🛠️ Tech Stack

**Frontend**

* HTML5
* CSS3 (Glassmorphism UI)
* JavaScript (Fetch API)

**Backend**

* Python (Django / Flask)
* OpenCV
* Face Recognition (`face_recognition`)
* Emotion Recognition Model

**Storage**

* `faces.pkl` → Stores face encodings for known users
* Database → Emotion logs and timestamps

---

## 📂 Project Structure

```
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── faces.pkl
├── views.py
├── emotion_model/
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Webcam captures live video frames.
2. Faces are detected and matched using stored encodings (`faces.pkl`).
3. Emotions are predicted for each detected face.
4. Emotion counts are stored and displayed in real time.
5. Confidence score is calculated dynamically from emotion statistics.
6. Final confidence is shown when counting stops.

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/face-emotion-recognition.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python manage.py runserver
```

4. Open browser:

```
http://127.0.0.1:8000/
```

---

## 🔐 Privacy & Security

* Camera access requires user permission
* Face data stored locally (`faces.pkl`)
* Emotion data can be deleted anytime via UI
* CSRF protection enabled for all requests

---

## 📌 Use Cases

* Online learning engagement analysis
* Corporate training feedback
* Mental health monitoring
* Driver alertness systems
* Smart classrooms
* Customer experience analytics

---

## 🔮 Future Improvements

* Multi-face tracking
* Emotion trend graphs
* Audio sentiment integration
* Cloud-based analytics dashboard
* Improved deep learning emotion models
* User-specific confidence calibration

---

## 👩‍💻 Author

**Punnavajhala Nagasrisreya**
Projects in **Machine Learning, Deep Learning & Computer Vision**

