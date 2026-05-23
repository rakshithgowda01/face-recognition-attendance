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

### Step 1: Register Students
![Register New Student](https://github.com/rakshithgowda01/face-recognition-attendance/blob/main/step1_register.png?raw=true)

The registration page allows you to add new students to the system. Upload 1 to 10 photographs of each student for better accuracy. The system uses these images to generate facial embeddings (numerical representations of faces).

**Photo Tips:**
- ✅ Clear front-facing face
- ✅ Good lighting
- ✅ Minimum 200×200 pixels
- ✅ JPG or PNG format
- ✅ Multiple angles help (increases recognition accuracy)
- ❌ Avoid sunglasses or masks
- ❌ No extreme side profiles (>45°)
- ❌ Avoid blurry or dark photos

---

### Step 2: View Dashboard
![Dashboard Overview](https://github.com/rakshithgowda01/face-recognition-attendance/blob/main/step2_dashboard.png?raw=true)

The dashboard provides a comprehensive overview of daily attendance statistics. It displays:
- **Registered**: Total number of registered students (5 in this example)
- **Present**: Students marked present today (2 present)
- **Absent**: Students not yet marked (3 absent)
- **Attendance Rate**: Overall percentage of students present (40.0%)

The attendance progress bar shows real-time visual feedback on the number of students who have been marked present. You can also manually mark students present or absent from here, and view the attendance details.

---

### Step 3: Live Attendance Marking
![Live Attendance](https://github.com/rakshithgowda01/face-recognition-attendance/blob/main/step3_live_attendance.png?raw=true)

The Live Attendance page uses your webcam to automatically recognize and mark students present. Simply start the webcam, and the ArcFace AI model will:
1. Detect faces in the webcam feed
2. Compare each face against registered student embeddings
3. Automatically mark matching students as present
4. Prevent duplicate marking (same student multiple times)

A live log on the right side displays all students who were successfully recognized, along with their timestamp and confidence score. The system is smart about preventing spam—once a student is marked, they won't be marked again until reset.

---

### Step 4: Manage Student Records
![All Students Management](https://github.com/rakshithgowda01/face-recognition-attendance/blob/main/step4_all_students.png?raw=true)

The All Students page provides a complete list of all registered students with their details:
- **Name**: Student's full name
- **Class/Section**: Student's batch or class identifier
- **Embeddings**: Number of facial embeddings generated from their registration photos
- **Student ID**: Unique identifier in the system

From this view, you can:
- 🔍 Search students by name or class
- 🔄 Refresh the student list
- 🗑️ Delete students from the system (if they withdraw or graduate)
- 📋 View and manage each student's facial data

---

## Key Technology: ArcFace

This system uses **ArcFace** (Additive Angular Margin for Deep Face Recognition), a state-of-the-art deep learning model that:
- Generates highly accurate facial embeddings (256-dimensional vectors)
- Provides robust face recognition across different angles, lighting, and expressions
- Works reliably for attendance marking in educational institutions
- Operates completely offline after initial model download

---

## File Structure

```
face-recognition-attendance/
├── app.py                 # Main application entry point
├── setup.py              # Model downloader and setup
├── requirements.txt      # Python dependencies
├── config.py             # Configuration settings
├── models/               # AI model storage
├── data/                 # Student embeddings and database
├── database/             # Attendance records database
├── attendance_logs/      # Daily attendance logs
├── assets/               # Screenshots and documentation images
└── README.md             # Documentation
```

---

## Troubleshooting

- **Camera not detected?** Ensure webcam is connected and permitted in Windows settings
- **Poor recognition?** Register with more photos from different angles and lighting conditions
- **Model download failed?** Check internet connection and retry `python setup.py`
- **Slow performance?** Close other applications to free up system resources
- **Images not showing?** Make sure images are in the repository root with correct filenames

---

## License

This project is open-source and available for educational purposes.

---

**Questions or Contributions?** Feel free to open an issue or submit a pull request!
