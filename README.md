# 🎓 Face Recognition Attendance System

AI-powered attendance system using ArcFace deep learning.
Recognizes registered students from webcam in real time and
marks attendance automatically.

Built by **Rakshith R** | BCA 1st Year | Gopalan College of Commerce

---

## Features

- Real-time face detection and recognition via webcam
- Register students with 1 to 10 photos (more = better accuracy)
- Manual attendance marking and unmarking from dashboard
- Auto attendance via webcam with duplicate prevention
- Export attendance to CSV and Excel
- Delete students from the system
- Works fully offline after first-time model download

---

## Quick Start (3 commands)

```bash
# 1. Clone the repo
git clone https://github.com/rakshithgowda01/face-recognition-attendance.git
cd face-recognition-attendance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download AI model (one time, ~300MB)
python setup.py

# 4. Run the app
python app.py
```

---

## Requirements

- Python 3.9, 3.10, or 3.11
- Windows (for the desktop GUI)
- Webcam (built-in or USB)
- Internet connection for first run (model download only)

---

## How It Works