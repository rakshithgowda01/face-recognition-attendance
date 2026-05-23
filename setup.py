"""
setup.py — Run this ONCE after cloning.
Downloads the ArcFace model (~300MB).
You never need internet again after this.
"""
import os, sys
from pathlib import Path

print("=" * 55)
print("  Face Recognition Attendance System — Setup")
print("=" * 55)
print()

# Check Python version
if sys.version_info < (3, 9):
    print("ERROR: Python 3.9 or higher required.")
    print(f"  You have: Python {sys.version}")
    print("  Download: https://python.org/downloads")
    sys.exit(1)
print(f"Python {sys.version_info.major}.{sys.version_info.minor} ✅")

# Create all folders
folders = [
    "data/raw_images",
    "data/augmented_images",
    "database",
    "models",
    "attendance_logs",
    "outputs",
]
for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
print("Folders created ✅")

# Download model
print()
print("Downloading ArcFace model (buffalo_l) — about 300MB")
print("This is a one-time download. Do not close this window.")
print()

try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        root="models",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print()
    print("Model downloaded and ready ✅")
except ImportError:
    print("ERROR: insightface not installed.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    print("Check your internet connection and try again.")
    sys.exit(1)

# Create empty database files
import pickle, json
db_dir = Path("database")

for fname, content in [
    ("embeddings.pkl",        {}),
    ("identity_profiles.pkl", {}),
    ("label_map.json",        {}),
]:
    p = db_dir / fname
    if not p.exists() or p.stat().st_size == 0:
        if fname.endswith(".pkl"):
            with open(p, "wb") as f:
                pickle.dump(content, f)
        else:
            with open(p, "w") as f:
                json.dump(content, f)
print("Database initialized ✅")

print()
print("=" * 55)
print("  Setup complete!")
print("  Run:  python app.py")
print("=" * 55)