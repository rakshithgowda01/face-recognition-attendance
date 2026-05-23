"""
Face Recognition Attendance System
Desktop App — Windows
Author  : Rakshith R | BCA 1st Year | Gopalan College of Commerce
GitHub  : github.com/yourusername/face-recognition-attendance
Run     : python app.py
"""

import os, sys, cv2, numpy as np, pandas as pd
import json, pickle, shutil, threading, time, csv
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
RAW_DIR      = BASE / "data" / "raw_images"
AUG_DIR      = BASE / "data" / "augmented_images"
DB_DIR       = BASE / "database"
MODELS_DIR   = BASE / "models"
ATT_DIR      = BASE / "attendance_logs"
OUT_DIR      = BASE / "outputs"
META_CSV     = BASE / "data" / "student_metadata.csv"
EMB_PATH     = DB_DIR / "embeddings.pkl"
PROF_PATH    = DB_DIR / "identity_profiles.pkl"
LMAP_PATH    = DB_DIR / "label_map.json"

THRESHOLD    = float(os.getenv("SIMILARITY_THRESHOLD", "0.38"))
TOP_K        = int(os.getenv("TOP_K", "5"))
N_AUGS       = int(os.getenv("NUM_AUGMENTS", "12"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MINUTES", "30"))
DATE_FMT     = "%Y-%m-%d"
TIME_FMT     = "%H:%M:%S"

for d in [RAW_DIR, AUG_DIR, DB_DIR, MODELS_DIR, ATT_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Utilities ──────────────────────────────────────────────────────────────────
def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def cos_sim(a, b): return float(np.dot(a, b))
def today():       return datetime.now().strftime(DATE_FMT)
def now_time():    return datetime.now().strftime(TIME_FMT)

# ── Augmentation ───────────────────────────────────────────────────────────────
def augment(img, n=None):
    if n is None: n = N_AUGS
    h, w = img.shape[:2]
    M1   = cv2.getRotationMatrix2D((w//2,h//2), -8, 1.0)
    M2   = cv2.getRotationMatrix2D((w//2,h//2),  8, 1.0)
    r1   = cv2.warpAffine(img, M1, (w,h), borderMode=cv2.BORDER_REFLECT)
    r2   = cv2.warpAffine(img, M2, (w,h), borderMode=cv2.BORDER_REFLECT)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = np.clip(hsv[:,:,0]+10, 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1]+20, 0, 255)
    hs   = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    nz   = np.clip(img.astype(np.int32) +
                   np.random.normal(0,8,img.shape).astype(np.int32),
                   0, 255).astype(np.uint8)
    m    = int(min(h,w) * 0.08)
    zm   = cv2.resize(img[m:h-m, m:w-m], (w,h))
    vs   = [
        img.copy(), cv2.flip(img, 1),
        np.clip(img.astype(np.int32)+40, 0,255).astype(np.uint8),
        np.clip(img.astype(np.int32)-40, 0,255).astype(np.uint8),
        np.clip(img.astype(np.float32)*1.3, 0,255).astype(np.uint8),
        np.clip(img.astype(np.float32)*0.75,0,255).astype(np.uint8),
        r1, r2,
        cv2.GaussianBlur(img,(3,3),0),
        cv2.filter2D(img,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])),
        hs, nz,
        np.clip(cv2.flip(img,1).astype(np.int32)+30,0,255).astype(np.uint8),
        np.clip(r1.astype(np.float32)*1.2,0,255).astype(np.uint8),
        zm,
    ]
    return vs[:n]

# ── Database ───────────────────────────────────────────────────────────────────
def db_load():
    def _pkl(p):
        try:
            if p.exists() and p.stat().st_size > 0:
                with open(p, "rb") as f: return pickle.load(f)
        except Exception: pass
        return {}
    def _js(p):
        try:
            if p.exists() and p.stat().st_size > 0:
                with open(p) as f: return json.load(f)
        except Exception: pass
        return {}
    return _pkl(EMB_PATH), _pkl(PROF_PATH), _js(LMAP_PATH)

def db_save(edb, pdb, lmap):
    with open(EMB_PATH, "wb") as f:
        pickle.dump(edb, f); f.flush(); os.fsync(f.fileno())
    with open(PROF_PATH, "wb") as f:
        pickle.dump(pdb, f); f.flush(); os.fsync(f.fileno())
    with open(LMAP_PATH, "w") as f:
        json.dump(lmap, f, indent=2); f.flush(); os.fsync(f.fileno())

# ── Embedder singleton ─────────────────────────────────────────────────────────
class Embedder:
    _inst = None

    def __init__(self):
        if not (MODELS_DIR / "models" / "buffalo_l").exists():
            raise FileNotFoundError(
                "Model not found.\nRun: python setup.py"
            )
        self.fa = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            providers=["CPUExecutionProvider"]
        )
        self.fa.prepare(ctx_id=-1, det_size=(640,640))

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def detect(self, bgr):
        return self.fa.get(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def embed_one(self, bgr):
        faces = self.detect(bgr)
        if not faces: return None
        return norm(max(faces, key=lambda f: f.det_score).embedding)

    def embed_all(self, bgr):
        return [{"bbox":       f.bbox.astype(int),
                 "embedding":  norm(f.embedding),
                 "det_score":  float(f.det_score)}
                for f in self.detect(bgr)]

    def embed_aug(self, imgs):
        embs, fail = [], 0
        for img in imgs:
            e = self.embed_one(img)
            if e is not None: embs.append(e)
            else: fail += 1
        return embs, fail

    def avg_prof(self, embs):
        return norm(np.mean(np.stack(embs), axis=0))

# ── Recognition ────────────────────────────────────────────────────────────────
def recog_one(q, edb, lmap):
    if not edb: return "Unknown", "", 0.0, ""
    best_id, best = None, -1.0
    for sid, el in edb.items():
        s = float(np.mean(
            sorted([cos_sim(q, e) for e in el], reverse=True)[:TOP_K]))
        if s > best: best, best_id = s, sid
    if best < THRESHOLD:
        return "Unknown", "", round(best*100, 1), ""
    info = lmap[best_id]
    return info["name"], info["class"], round(best*100, 1), best_id

def recog_frame(faces, edb, lmap):
    return [{**f,
             **dict(zip(["name","class","confidence","student_id"],
                        recog_one(f["embedding"], edb, lmap)))}
            for f in faces]

# ── Attendance ─────────────────────────────────────────────────────────────────
def att_file(ds=None):
    if ds is None: ds = today()
    p = ATT_DIR / f"attendance_{ds}.csv"
    if not p.exists():
        pd.DataFrame(columns=["student_id","name","class",
                               "date","time","confidence",
                               "method"]).to_csv(p, index=False)
    return p

def load_att(ds=None):
    try:
        df = pd.read_csv(att_file(ds))
        if "method" not in df.columns:
            df["method"] = "auto"
        return df if not df.empty else pd.DataFrame(
            columns=["student_id","name","class",
                     "date","time","confidence","method"])
    except Exception:
        return pd.DataFrame(
            columns=["student_id","name","class",
                     "date","time","confidence","method"])

def can_mark(sid, df):
    if df.empty or sid not in df["student_id"].values:
        return True
    try:
        last = df[df["student_id"]==sid].iloc[-1]["time"]
        t    = datetime.strptime(f"{today()} {last}",
                                 f"{DATE_FMT} {TIME_FMT}")
        return (datetime.now() - t) >= timedelta(minutes=COOLDOWN_MIN)
    except Exception:
        return True

def mark_att(sid, name, cls, conf, method="auto"):
    df = load_att()
    if not can_mark(sid, df): return False
    with open(att_file(), "a", newline="") as f:
        csv.writer(f).writerow(
            [sid, name, cls, today(), now_time(), f"{conf}%", method])
    return True

def unmark_att(sid):
    """Remove today's attendance entry for a student."""
    p  = att_file()
    df = load_att()
    if df.empty or sid not in df["student_id"].values:
        return False
    df = df[df["student_id"] != sid]
    df.to_csv(p, index=False)
    return True

def export_excel():
    df  = load_att()
    out = ATT_DIR / f"attendance_{today()}.xlsx"
    _, _, lmap = db_load()
    pids = set(df["student_id"].unique()) if not df.empty else set()
    summary = []
    for sid, info in lmap.items():
        is_present = sid in pids
        t = (df[df["student_id"]==sid].iloc[0]["time"]
             if not df.empty and sid in df["student_id"].values else "-")
        m = (df[df["student_id"]==sid].iloc[0]["method"]
             if not df.empty and sid in df["student_id"].values else "-")
        summary.append({
            "Name":   info["name"], "Class": info["class"],
            "Status": "Present" if is_present else "Absent",
            "Time":   t, "Method": m
        })
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Attendance")
        pd.DataFrame(summary).to_excel(w, index=False, sheet_name="Summary")
    return out

# ── Registration (supports multiple photos) ────────────────────────────────────
def register_student(name, class_name, image_paths, progress_cb=None):
    """
    Register a student using one OR more photos.
    image_paths: list of file paths (str or Path)
    More photos = more embeddings = better accuracy.
    """
    sid  = f"student_{name.replace(' ','_').lower()}"
    emb  = Embedder.get()
    all_embs = []

    # Load existing embeddings for this student if re-registering
    edb, pdb, lmap = db_load()
    existing_embs  = edb.get(sid, [])

    for idx, img_path in enumerate(image_paths):
        prefix = f"Photo {idx+1}/{len(image_paths)}: "

        if progress_cb:
            progress_cb(f"{prefix}Reading image...")
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")

        if progress_cb:
            progress_cb(f"{prefix}Detecting face...")
        faces = emb.detect(img)
        if not faces:
            raise ValueError(
                f"No face in photo {idx+1}. "
                "Use a clear front-facing photo.")
        score = max(f.det_score for f in faces)
        if score < 0.5:
            raise ValueError(
                f"Photo {idx+1} quality too low "
                f"(score {score:.2f}). Use a better photo.")

        # Save raw image
        raw = RAW_DIR / sid
        raw.mkdir(parents=True, exist_ok=True)
        ext = Path(img_path).suffix
        shutil.copy2(str(img_path),
                     str(raw / f"photo_{idx+1:02d}{ext}"))

        if progress_cb:
            progress_cb(f"{prefix}Augmenting ({N_AUGS} variants)...")
        aug_imgs = augment(img)
        aug_dir  = AUG_DIR / sid
        aug_dir.mkdir(parents=True, exist_ok=True)
        offset   = idx * N_AUGS
        for i, a in enumerate(aug_imgs):
            cv2.imwrite(str(aug_dir / f"aug_{offset+i+1:03d}.jpg"), a)

        if progress_cb:
            progress_cb(f"{prefix}Generating embeddings...")
        embs, fail = emb.embed_aug(aug_imgs)
        if progress_cb:
            progress_cb(
                f"{prefix}Got {len(embs)} embeddings "
                f"({fail} skipped).")
        all_embs.extend(embs)

    # Merge with any existing embeddings (if adding more photos later)
    merged_embs = existing_embs + all_embs
    if len(merged_embs) < 3:
        raise ValueError(
            f"Only {len(merged_embs)} valid embeddings total. "
            "Use clearer photos.")

    if progress_cb:
        progress_cb(
            f"Computing identity profile from "
            f"{len(merged_embs)} embeddings...")
    profile = emb.avg_prof(merged_embs)

    # Save
    if progress_cb: progress_cb("Saving to database...")
    edb[sid]  = merged_embs
    pdb[sid]  = profile
    lmap[sid] = {"name": name, "class": class_name}
    db_save(edb, pdb, lmap)

    # Metadata CSV
    row = {"student_id": sid, "name": name,
           "class": class_name, "date": today(),
           "photos": len(image_paths),
           "embeddings": len(merged_embs)}
    if META_CSV.exists() and META_CSV.stat().st_size > 0:
        try:
            df = pd.read_csv(META_CSV)
            df = df[df["student_id"] != sid]
            df = pd.concat([df, pd.DataFrame([row])],
                            ignore_index=True)
        except Exception:
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
    df.to_csv(META_CSV, index=False)

    if progress_cb:
        progress_cb(f"Done! {len(merged_embs)} total embeddings.")
    return sid, len(merged_embs)

def delete_student(sid):
    """Completely remove a student from the system."""
    edb, pdb, lmap = db_load()
    for d in [edb, pdb, lmap]:
        d.pop(sid, None)
    db_save(edb, pdb, lmap)
    # Remove images
    for folder in [RAW_DIR / sid, AUG_DIR / sid]:
        if folder.exists():
            shutil.rmtree(str(folder))
    # Remove from metadata
    if META_CSV.exists():
        try:
            df = pd.read_csv(META_CSV)
            df = df[df["student_id"] != sid]
            df.to_csv(META_CSV, index=False)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT  = "#7c6ef5"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
YELLOW  = "#f39c12"
CARD    = "#141824"
BORDER  = "#1e2236"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance System — Rakshith R")
        self.geometry("1150x720")
        self.minsize(960, 620)
        self.configure(fg_color="#0d0f18")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # State
        self._model_ready  = False
        self._cam_running  = False
        self._cap          = None
        self._cam_thread   = None
        self._last_results = []
        self._att_log      = set()
        self._reg_photos   = []       # list of selected photo paths
        self._edb = self._pdb = self._lmap = {}

        self._build_ui()
        self._load_model_bg()

    # ── Build UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(
            self, width=210, corner_radius=0,
            fg_color=CARD, border_width=1, border_color=BORDER)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        ctk.CTkLabel(
            self.sidebar, text="🎓  FaceAttend",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=ACCENT
        ).pack(pady=(24,2), padx=16, anchor="w")
        ctk.CTkLabel(
            self.sidebar, text="Gopalan College of Commerce",
            font=ctk.CTkFont(size=9), text_color="gray"
        ).pack(pady=(0,20), padx=16, anchor="w")

        # Nav buttons
        self._nav = {}
        for key, icon, label in [
            ("dashboard",  "📊", "Dashboard"),
            ("attendance", "📷", "Live Attendance"),
            ("register",   "➕", "Register Student"),
            ("records",    "📋", "Attendance Records"),
            ("students",   "👥", "All Students"),
        ]:
            btn = ctk.CTkButton(
                self.sidebar,
                text=f"  {icon}  {label}",
                anchor="w",
                fg_color="transparent",
                hover_color="#1e2236",
                font=ctk.CTkFont(size=12),
                corner_radius=8,
                command=lambda k=key: self._switch(k)
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav[key] = btn

        # Model status at bottom of sidebar
        self.model_lbl = ctk.CTkLabel(
            self.sidebar,
            text="⏳  Loading model...",
            font=ctk.CTkFont(size=9),
            text_color="orange"
        )
        self.model_lbl.pack(side="bottom", pady=14, padx=14, anchor="w")

        # Main content area
        self.main = ctk.CTkFrame(
            self, corner_radius=0, fg_color="#0d0f18")
        self.main.pack(side="left", fill="both", expand=True)

        # Build all pages
        self._pages = {
            "dashboard":  self._page_dashboard(),
            "attendance": self._page_attendance(),
            "register":   self._page_register(),
            "records":    self._page_records(),
            "students":   self._page_students(),
        }

        self._switch("dashboard")

    def _switch(self, key):
        for p in self._pages.values():
            p.pack_forget()
        self._pages[key].pack(
            fill="both", expand=True, padx=22, pady=20)
        for k, btn in self._nav.items():
            btn.configure(
                fg_color="#1e2236" if k == key else "transparent",
                text_color=ACCENT if k == key else "white"
            )
        if key == "dashboard":  self._refresh_dashboard()
        if key == "records":    self._refresh_records()
        if key == "students":   self._refresh_students()

    # ── Shared widget helpers ──────────────────────────────────────────────────
    def _card(self, parent, **kw):
        return ctk.CTkFrame(
            parent, corner_radius=12,
            fg_color=CARD,
            border_width=1, border_color=BORDER, **kw)

    def _section(self, parent, title):
        ctk.CTkLabel(
            parent, text=title,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        ).pack(anchor="w", padx=16, pady=(14,4))

    def _stat_card(self, parent, label, color):
        card = self._card(parent)
        card.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(
            card, text=label,
            font=ctk.CTkFont(size=9),
            text_color="gray"
        ).pack(pady=(12,2))
        val = ctk.CTkLabel(
            card, text="—",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=color)
        val.pack(pady=(0,12))
        return val

    # ── DASHBOARD PAGE ─────────────────────────────────────────────────────────
    def _page_dashboard(self):
        page = ctk.CTkFrame(self.main, fg_color="transparent")

        # Title
        ctk.CTkLabel(
            page, text="Dashboard",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        self._dash_sub = ctk.CTkLabel(
            page, text="Today",
            font=ctk.CTkFont(size=11), text_color="gray")
        self._dash_sub.pack(anchor="w", pady=(0,14))

        # Stat row
        row = ctk.CTkFrame(page, fg_color="transparent")
        row.pack(fill="x", pady=(0,14))
        self._d_reg  = self._stat_card(row, "Registered", ACCENT)
        self._d_pre  = self._stat_card(row, "Present",    GREEN)
        self._d_abs  = self._stat_card(row, "Absent",     RED)
        self._d_rate = self._stat_card(row, "Rate",       YELLOW)

        # Progress bar
        prog_card = self._card(page)
        prog_card.pack(fill="x", pady=(0,14))
        self._section(prog_card, "ATTENDANCE PROGRESS")
        self._prog_bar = ctk.CTkProgressBar(
            prog_card, height=10,
            progress_color=ACCENT)
        self._prog_bar.pack(fill="x", padx=16, pady=(0,4))
        self._prog_bar.set(0)
        self._prog_lbl = ctk.CTkLabel(
            prog_card, text="—",
            font=ctk.CTkFont(size=10), text_color="gray")
        self._prog_lbl.pack(padx=16, anchor="w", pady=(0,12))

        # Two column: present | absent
        cols = ctk.CTkFrame(page, fg_color="transparent")
        cols.pack(fill="both", expand=True)

        # Present table
        left = self._card(cols)
        left.pack(side="left", fill="both", expand=True, padx=(0,7))
        hdr_l = ctk.CTkFrame(left, fg_color="#1a1d36", corner_radius=0)
        hdr_l.pack(fill="x")
        ctk.CTkLabel(
            hdr_l, text="✅  Present Today",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=GREEN
        ).pack(side="left", padx=14, pady=8)
        ctk.CTkButton(
            hdr_l, text="↻", width=30,
            fg_color="transparent", hover_color=BORDER,
            command=self._refresh_dashboard
        ).pack(side="right", padx=8)
        self._present_scroll = ctk.CTkScrollableFrame(
            left, fg_color="transparent")
        self._present_scroll.pack(fill="both", expand=True)

        # Absent table with manual mark
        right = self._card(cols)
        right.pack(side="right", fill="both", expand=True, padx=(7,0))
        hdr_r = ctk.CTkFrame(right, fg_color="#2b0d0d", corner_radius=0)
        hdr_r.pack(fill="x")
        ctk.CTkLabel(
            hdr_r, text="❌  Absent Today",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=RED
        ).pack(side="left", padx=14, pady=8)
        self._absent_scroll = ctk.CTkScrollableFrame(
            right, fg_color="transparent")
        self._absent_scroll.pack(fill="both", expand=True)

        # Export button
        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.pack(fill="x", pady=(12,0))
        ctk.CTkButton(
            btn_row, text="⬇  Export Excel",
            fg_color=GREEN, hover_color="#27ae60",
            text_color="black", width=150,
            command=self._do_export
        ).pack(side="left", padx=(0,8))
        ctk.CTkButton(
            btn_row, text="📂  Open Folder",
            fg_color="transparent",
            border_width=1, border_color=BORDER,
            width=130,
            command=lambda: os.startfile(str(ATT_DIR))
        ).pack(side="left")

        return page

    def _refresh_dashboard(self):
        edb, pdb, lmap = db_load()
        self._edb, self._pdb, self._lmap = edb, pdb, lmap
        df   = load_att()
        pids = set(df["student_id"].unique()) if not df.empty else set()
        tot  = len(lmap)
        pre  = len(pids)
        ab   = tot - pre
        rate = round(pre/tot*100, 1) if tot else 0

        self._d_reg.configure( text=str(tot))
        self._d_pre.configure( text=str(pre))
        self._d_abs.configure( text=str(ab))
        self._d_rate.configure(text=f"{rate}%")
        self._prog_bar.set(rate/100)
        self._prog_lbl.configure(
            text=f"{pre} of {tot} students marked present")
        self._dash_sub.configure(
            text=f"Today — {today()}")

        # Present list
        for w in self._present_scroll.winfo_children():
            w.destroy()
        present_sids = [s for s in pids if s in lmap]
        if not present_sids:
            ctk.CTkLabel(
                self._present_scroll,
                text="No one marked yet.",
                text_color="gray",
                font=ctk.CTkFont(size=11)
            ).pack(pady=16)
        else:
            for sid in present_sids:
                info = lmap[sid]
                rec  = df[df["student_id"]==sid].iloc[0]
                row  = ctk.CTkFrame(
                    self._present_scroll,
                    fg_color="#0d2b1a", corner_radius=8)
                row.pack(fill="x", pady=2, padx=4)
                ctk.CTkLabel(
                    row,
                    text=f"  {info['name']}",
                    font=ctk.CTkFont(weight="bold"),
                    width=160, anchor="w"
                ).pack(side="left", padx=(4,0), pady=6)
                ctk.CTkLabel(
                    row,
                    text=info["class"],
                    font=ctk.CTkFont(size=10),
                    text_color="gray", width=70
                ).pack(side="left")
                ctk.CTkLabel(
                    row,
                    text=rec["time"],
                    font=ctk.CTkFont(size=10),
                    text_color=GREEN, width=70
                ).pack(side="left")
                # Unmark button
                ctk.CTkButton(
                    row, text="✕ Unmark",
                    width=80, height=26,
                    fg_color="transparent",
                    border_width=1, border_color=RED,
                    text_color=RED,
                    font=ctk.CTkFont(size=10),
                    command=lambda s=sid, n=info["name"]:
                        self._do_unmark(s, n)
                ).pack(side="right", padx=6)

        # Absent list with Mark Present button
        for w in self._absent_scroll.winfo_children():
            w.destroy()
        absent_sids = [s for s in lmap if s not in pids]
        if not absent_sids:
            ctk.CTkLabel(
                self._absent_scroll,
                text="🎉  Everyone is present!",
                text_color=GREEN,
                font=ctk.CTkFont(size=11)
            ).pack(pady=16)
        else:
            for sid in absent_sids:
                info = lmap[sid]
                row  = ctk.CTkFrame(
                    self._absent_scroll,
                    fg_color="#1a1214", corner_radius=8)
                row.pack(fill="x", pady=2, padx=4)
                ctk.CTkLabel(
                    row,
                    text=f"  {info['name']}",
                    font=ctk.CTkFont(weight="bold"),
                    width=160, anchor="w"
                ).pack(side="left", padx=(4,0), pady=6)
                ctk.CTkLabel(
                    row,
                    text=info["class"],
                    font=ctk.CTkFont(size=10),
                    text_color="gray", width=70
                ).pack(side="left")
                # Mark Present button
                ctk.CTkButton(
                    row,
                    text="✅ Mark Present",
                    width=120, height=26,
                    fg_color=GREEN,
                    hover_color="#27ae60",
                    text_color="black",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda s=sid,
                                   n=info["name"],
                                   c=info["class"]:
                        self._do_manual_mark(s, n, c)
                ).pack(side="right", padx=6)

    def _do_manual_mark(self, sid, name, cls):
        marked = mark_att(sid, name, cls, 100, method="manual")
        if marked:
            messagebox.showinfo(
                "Marked",
                f"✅ {name} marked as Present.\n"
                f"Method: Manual\nTime: {now_time()}")
            self._refresh_dashboard()
        else:
            messagebox.showinfo(
                "Already Marked",
                f"{name} was already marked within "
                f"the last {COOLDOWN_MIN} minutes.")

    def _do_unmark(self, sid, name):
        if messagebox.askyesno(
            "Unmark Attendance",
            f"Remove today's attendance for {name}?"):
            unmark_att(sid)
            self._refresh_dashboard()

    def _do_export(self):
        try:
            out = export_excel()
            messagebox.showinfo(
                "Exported",
                f"Excel saved:\n{out}\n\nOpening folder...")
            os.startfile(str(ATT_DIR))
        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    # ── ATTENDANCE PAGE ────────────────────────────────────────────────────────
    def _page_attendance(self):
        page = ctk.CTkFrame(self.main, fg_color="transparent")

        ctk.CTkLabel(
            page, text="Live Attendance",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            page,
            text="Start webcam — attendance marks automatically",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(anchor="w", pady=(0,14))

        content = ctk.CTkFrame(page, fg_color="transparent")
        content.pack(fill="both", expand=True)

        # Left — webcam
        left = self._card(content)
        left.pack(side="left", fill="both", expand=True, padx=(0,10))

        self._cam_lbl = ctk.CTkLabel(
            left, text="Webcam not started",
            width=460, height=345,
            fg_color="#080a10", corner_radius=10,
            font=ctk.CTkFont(size=13), text_color="gray"
        )
        self._cam_lbl.pack(padx=14, pady=(14,8))

        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.pack(fill="x", padx=14, pady=(0,14))

        self._cam_btn = ctk.CTkButton(
            ctrl, text="▶  Start Webcam",
            width=160, height=38,
            fg_color=GREEN, hover_color="#27ae60",
            text_color="black",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_cam
        )
        self._cam_btn.pack(side="left", padx=(0,10))

        self._cam_status = ctk.CTkLabel(
            ctrl, text="● Stopped",
            text_color=RED,
            font=ctk.CTkFont(size=12))
        self._cam_status.pack(side="left")

        # Right — live log
        right = self._card(content)
        right.pack(side="right", fill="y", ipadx=4)
        right.configure(width=280)
        right.pack_propagate(False)

        ctk.CTkLabel(
            right, text="Live Log",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(14,2), padx=14, anchor="w")
        self._live_count = ctk.CTkLabel(
            right, text="0 marked",
            text_color=GREEN,
            font=ctk.CTkFont(size=10))
        self._live_count.pack(padx=14, anchor="w", pady=(0,6))

        self._live_scroll = ctk.CTkScrollableFrame(
            right, fg_color="transparent")
        self._live_scroll.pack(
            fill="both", expand=True, padx=6, pady=(0,10))

        return page

    def _toggle_cam(self):
        if self._cam_running: self._stop_cam()
        else: self._start_cam()

    def _start_cam(self):
        if not self._model_ready:
            messagebox.showwarning("Wait", "Model still loading.")
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Cannot open webcam.\n\n"
                "• Check no other app is using it\n"
                "• Try unplugging and replugging\n"
                "• Try changing VideoCapture(0) → (1)")
            return
        self._edb, self._pdb, self._lmap = db_load()
        self._cam_running = True
        self._att_log     = set()
        self._cam_btn.configure(
            text="⏹  Stop Webcam",
            fg_color=RED, hover_color="#c0392b",
            text_color="white")
        self._cam_status.configure(
            text="● Live", text_color=GREEN)
        self._cam_thread = threading.Thread(
            target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    def _stop_cam(self):
        self._cam_running = False
        time.sleep(0.2)
        if self._cap: self._cap.release()
        self._cam_btn.configure(
            text="▶  Start Webcam",
            fg_color=GREEN, hover_color="#27ae60",
            text_color="black")
        self._cam_status.configure(
            text="● Stopped", text_color=RED)
        self._cam_lbl.configure(image=None, text="Webcam stopped")
        try:
            out = export_excel()
            self.after(200, lambda: messagebox.showinfo(
                "Session Saved",
                f"{len(self._att_log)} student(s) marked.\n"
                f"Excel: {out}"))
        except Exception:
            pass

    def _cam_loop(self):
        n, last = 0, []
        emb = Embedder.get()
        while self._cam_running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            n += 1
            if n % 4 == 0:
                try:
                    faces = emb.embed_all(frame)
                    if faces:
                        last = recog_frame(
                            faces, self._edb, self._lmap)
                        for r in last:
                            if (r["name"] != "Unknown"
                                    and r["student_id"]
                                    and r["student_id"]
                                    not in self._att_log):
                                if mark_att(
                                    r["student_id"],
                                    r["name"],
                                    r["class"],
                                    r["confidence"],
                                    method="webcam"
                                ):
                                    self._att_log.add(
                                        r["student_id"])
                                    self.after(
                                        0,
                                        self._add_live_log,
                                        r["name"],
                                        r["class"],
                                        r["confidence"])
                    else:
                        last = []
                except Exception:
                    last = []

            display = frame.copy()
            for r in last:
                x1,y1,x2,y2 = r["bbox"]
                col = (0,200,0) if r["name"]!="Unknown" \
                      else (0,0,220)
                cv2.rectangle(display,(x1,y1),(x2,y2),col,2)
                label = (f"{r['name']} {r['confidence']}%"
                         if r["name"]!="Unknown" else "Unknown")
                cv2.putText(
                    display, label,
                    (x1, max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, col, 2)
            cv2.putText(
                display,
                f"Marked: {len(self._att_log)}",
                (10,28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,220,220), 2)

            rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((460,345))
            ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
            self.after(0, self._update_frame, ctk_img)
            self.after(0, lambda:
                self._live_count.configure(
                    text=f"{len(self._att_log)} marked today"))

    def _update_frame(self, img):
        self._cam_lbl.configure(image=img, text="")
        self._cam_lbl._image = img

    def _add_live_log(self, name, cls, conf):
        row = ctk.CTkFrame(
            self._live_scroll,
            fg_color="#0d2b1a", corner_radius=7)
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row,
            text=f"✅  {name}",
            font=ctk.CTkFont(weight="bold"),
            text_color=GREEN
        ).pack(anchor="w", padx=8, pady=(5,0))
        ctk.CTkLabel(
            row,
            text=f"   {cls}  •  {conf}%  •  {now_time()}",
            font=ctk.CTkFont(size=9), text_color="gray"
        ).pack(anchor="w", padx=8, pady=(0,5))

    # ── REGISTER PAGE ──────────────────────────────────────────────────────────
    def _page_register(self):
        page = ctk.CTkFrame(self.main, fg_color="transparent")

        ctk.CTkLabel(
            page, text="Register New Student",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            page,
            text="Upload 1 to 10 photos — more photos = better accuracy",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(anchor="w", pady=(0,14))

        content = ctk.CTkFrame(page, fg_color="transparent")
        content.pack(fill="both", expand=True)

        # Left — form
        left = self._card(content)
        left.pack(side="left", fill="both", expand=True, padx=(0,10))

        form = ctk.CTkScrollableFrame(
            left, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=4)

        # Name
        self._section(form, "STUDENT DETAILS")
        ctk.CTkLabel(
            form, text="Full Name *",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=12, pady=(0,3))
        self._reg_name = ctk.CTkEntry(
            form, placeholder_text="e.g. Rahul Sharma",
            height=36)
        self._reg_name.pack(fill="x", padx=12, pady=(0,10))

        ctk.CTkLabel(
            form, text="Class / Section *",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=12, pady=(0,3))
        self._reg_cls = ctk.CTkEntry(
            form, placeholder_text="e.g. CSE-A",
            height=36)
        self._reg_cls.pack(fill="x", padx=12, pady=(0,14))

        # Photos
        self._section(form, "PHOTOS (1 to 10)")
        ctk.CTkLabel(
            form,
            text="Select one or more photos of the student.\n"
                 "Different angles and lighting = better recognition.",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            justify="left"
        ).pack(anchor="w", padx=12, pady=(0,8))

        # Upload button
        ctk.CTkButton(
            form,
            text="📷  Select Photos (hold Ctrl for multiple)",
            height=44,
            fg_color="#1a1d36",
            hover_color="#2a2d4e",
            border_width=2, border_color=ACCENT,
            command=self._pick_photos
        ).pack(fill="x", padx=12, pady=(0,6))

        # Photo count label
        self._photo_count_lbl = ctk.CTkLabel(
            form, text="No photos selected",
            font=ctk.CTkFont(size=10), text_color="gray")
        self._photo_count_lbl.pack(padx=12, anchor="w")

        # Photo preview strip
        self._preview_strip = ctk.CTkFrame(
            form, fg_color="transparent", height=90)
        self._preview_strip.pack(
            fill="x", padx=12, pady=(6,12))

        # Progress
        self._reg_prog = ctk.CTkProgressBar(
            form, height=8, progress_color=ACCENT)
        self._reg_prog.pack(fill="x", padx=12, pady=(0,4))
        self._reg_prog.set(0)
        self._reg_status = ctk.CTkLabel(
            form, text="Ready",
            font=ctk.CTkFont(size=9), text_color="gray")
        self._reg_status.pack(padx=12, anchor="w", pady=(0,10))

        self._reg_btn = ctk.CTkButton(
            form,
            text="Register Student",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._do_register
        )
        self._reg_btn.pack(fill="x", padx=12, pady=(0,16))

        # Right — tips
        right = ctk.CTkFrame(content, width=250)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tips = self._card(right)
        tips.pack(fill="x", padx=8, pady=(0,10))

        ctk.CTkLabel(
            tips, text="Photo Tips",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT
        ).pack(pady=(12,6), padx=12)

        for t in [
            "✅  Clear front-facing face",
            "✅  Good lighting",
            "✅  Min 200×200 px",
            "✅  JPG or PNG",
            "✅  Multiple angles helps",
            "❌  Sunglasses or mask",
            "❌  Side profile > 45°",
            "❌  Blurry or dark photos",
        ]:
            ctk.CTkLabel(
                tips, text=t,
                font=ctk.CTkFont(size=10),
                text_color="gray", anchor="w"
            ).pack(anchor="w", padx=12, pady=1)
        ctk.CTkLabel(tips, text="").pack(pady=4)

        tip2 = self._card(right)
        tip2.pack(fill="x", padx=8)
        ctk.CTkLabel(
            tip2, text="💡 Pro Tip",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=YELLOW
        ).pack(pady=(10,4), padx=12)
        ctk.CTkLabel(
            tip2,
            text="Take registration photos\nwith the same webcam\nyou use for attendance.\nThis gives best results.",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            justify="left"
        ).pack(padx=12, pady=(0,12))

        return page

    def _pick_photos(self):
        paths = filedialog.askopenfilenames(
            title="Select Student Photos (hold Ctrl for multiple)",
            filetypes=[("Images",
                        "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not paths: return
        if len(paths) > 10:
            messagebox.showwarning(
                "Too Many",
                "Maximum 10 photos per registration.\n"
                f"First 10 selected will be used.")
            paths = paths[:10]

        self._reg_photos = list(paths)
        n = len(self._reg_photos)
        self._photo_count_lbl.configure(
            text=f"{n} photo{'s' if n>1 else ''} selected",
            text_color=GREEN)

        # Show thumbnail strip
        for w in self._preview_strip.winfo_children():
            w.destroy()
        for path in self._reg_photos[:6]:
            try:
                img = Image.open(path)
                img.thumbnail((70, 70))
                ctk_img = ctk.CTkImage(img, size=(70,70))
                lbl = ctk.CTkLabel(
                    self._preview_strip,
                    image=ctk_img, text="",
                    width=70, height=70)
                lbl._image = ctk_img
                lbl.pack(side="left", padx=3)
            except Exception:
                pass
        if n > 6:
            ctk.CTkLabel(
                self._preview_strip,
                text=f"+{n-6} more",
                text_color="gray",
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=4)

    def _do_register(self):
        name = self._reg_name.get().strip()
        cls  = self._reg_cls.get().strip()
        if not name:
            messagebox.showwarning(
                "Missing", "Enter student name.")
            return
        if not cls:
            messagebox.showwarning(
                "Missing", "Enter class/section.")
            return
        if not self._reg_photos:
            messagebox.showwarning(
                "Missing",
                "Select at least one photo.")
            return
        if not self._model_ready:
            messagebox.showwarning(
                "Wait", "Model still loading.")
            return

        self._reg_btn.configure(
            state="disabled",
            text="Registering...")
        self._reg_prog.set(0)

        total   = len(self._reg_photos)
        counter = [0]

        def progress_cb(msg):
            counter[0] = min(counter[0] + 1, total * 7)
            val = counter[0] / (total * 7)
            self.after(0, self._reg_prog.set, val)
            self.after(0,
                self._reg_status.configure, {"text": msg})

        def run():
            try:
                sid, n_embs = register_student(
                    name, cls,
                    self._reg_photos,
                    progress_cb)
                self.after(
                    0, self._on_reg_ok, name, n_embs, total)
            except Exception as e:
                self.after(0, self._on_reg_fail, str(e))

        threading.Thread(target=run, daemon=True).start()

    def _on_reg_ok(self, name, n_embs, n_photos):
        self._reg_btn.configure(
            state="normal", text="Register Student")
        self._reg_prog.set(1)
        self._reg_status.configure(
            text=f"✅  Registered from {n_photos} photo(s),"
                 f" {n_embs} embeddings",
            text_color=GREEN)
        messagebox.showinfo(
            "Registered",
            f"✅  {name} registered!\n\n"
            f"Photos used   : {n_photos}\n"
            f"Embeddings    : {n_embs}\n\n"
            "Recognition is ready immediately.")
        self._reg_name.delete(0, "end")
        self._reg_cls.delete(0, "end")
        self._reg_photos = []
        self._photo_count_lbl.configure(
            text="No photos selected", text_color="gray")
        for w in self._preview_strip.winfo_children():
            w.destroy()

    def _on_reg_fail(self, err):
        self._reg_btn.configure(
            state="normal", text="Register Student")
        self._reg_prog.set(0)
        self._reg_status.configure(
            text=f"❌  {err}", text_color=RED)
        messagebox.showerror("Registration Failed", err)

    # ── RECORDS PAGE ───────────────────────────────────────────────────────────
    def _page_records(self):
        page = ctk.CTkFrame(self.main, fg_color="transparent")

        ctk.CTkLabel(
            page, text="Attendance Records",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            page, text="Full records with manual mark option",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(anchor="w", pady=(0,14))

        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0,12))

        ctk.CTkButton(
            btn_row, text="↻  Refresh",
            width=110, command=self._refresh_records
        ).pack(side="left", padx=(0,8))
        ctk.CTkButton(
            btn_row, text="⬇  Export Excel",
            width=140, fg_color=GREEN,
            hover_color="#27ae60", text_color="black",
            command=self._do_export
        ).pack(side="left", padx=(0,8))
        ctk.CTkButton(
            btn_row, text="📂  Open Folder",
            width=130, fg_color="transparent",
            border_width=1, border_color=BORDER,
            command=lambda: os.startfile(str(ATT_DIR))
        ).pack(side="left")

        # Table
        self._rec_scroll = ctk.CTkScrollableFrame(page)
        self._rec_scroll.pack(fill="both", expand=True)

        return page

    def _refresh_records(self):
        for w in self._rec_scroll.winfo_children():
            w.destroy()
        edb, pdb, lmap = db_load()
        df   = load_att()
        pids = set(df["student_id"].unique()) \
               if not df.empty else set()

        # Header
        hdr = ctk.CTkFrame(
            self._rec_scroll, fg_color="#1a1d36",
            corner_radius=8)
        hdr.pack(fill="x", pady=(0,4))
        for col, w in [
            ("Name",170), ("Class",90), ("Status",90),
            ("Time",90),  ("Conf",80),  ("Method",90),
            ("Action",120)
        ]:
            ctk.CTkLabel(
                hdr, text=col, width=w,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=ACCENT
            ).pack(side="left", padx=6, pady=8)

        for sid, info in lmap.items():
            is_p = sid in pids
            row  = ctk.CTkFrame(
                self._rec_scroll,
                fg_color="#0d2b1a" if is_p else "#1a1214",
                corner_radius=7)
            row.pack(fill="x", pady=2)

            if is_p and not df.empty:
                rec  = df[df["student_id"]==sid].iloc[0]
                t    = rec["time"]
                conf = rec.get("confidence","—")
                meth = rec.get("method","auto")
                scol = GREEN
                stxt = "Present"
            else:
                t    = "—"
                conf = "—"
                meth = "—"
                scol = RED
                stxt = "Absent"

            for val, w, col in [
                (info["name"],  170, None),
                (info["class"],  90, None),
                (stxt,           90, scol),
                (t,              90, None),
                (conf,           80, None),
                (meth,           90, "gray"),
            ]:
                kw = {"text":val,"width":w,"font":ctk.CTkFont(size=10)}
                if col: kw["text_color"] = col
                ctk.CTkLabel(row,**kw).pack(
                    side="left",padx=6,pady=7)

            # Action button
            if is_p:
                ctk.CTkButton(
                    row, text="✕ Unmark",
                    width=100, height=26,
                    fg_color="transparent",
                    border_width=1, border_color=RED,
                    text_color=RED,
                    font=ctk.CTkFont(size=9),
                    command=lambda s=sid, n=info["name"]:
                        self._do_unmark(s,n)
                ).pack(side="left", padx=6)
            else:
                ctk.CTkButton(
                    row, text="✅ Mark Present",
                    width=110, height=26,
                    fg_color=GREEN,
                    hover_color="#27ae60",
                    text_color="black",
                    font=ctk.CTkFont(size=9, weight="bold"),
                    command=lambda s=sid,
                                   n=info["name"],
                                   c=info["class"]:
                        self._do_manual_mark_refresh(s,n,c)
                ).pack(side="left", padx=6)

    def _do_manual_mark_refresh(self, sid, name, cls):
        marked = mark_att(sid, name, cls, 100, method="manual")
        if marked:
            self._refresh_records()
        else:
            messagebox.showinfo(
                "Already Marked",
                f"{name} was recently marked.")

    # ── STUDENTS PAGE ──────────────────────────────────────────────────────────
    def _page_students(self):
        page = ctk.CTkFrame(self.main, fg_color="transparent")

        ctk.CTkLabel(
            page, text="All Students",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            page, text="Registered students and their data",
            font=ctk.CTkFont(size=11), text_color="gray"
        ).pack(anchor="w", pady=(0,14))

        btn_row = ctk.CTkFrame(page, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0,10))
        ctk.CTkButton(
            btn_row, text="↻  Refresh",
            width=110, command=self._refresh_students
        ).pack(side="left", padx=(0,8))

        # Search
        self._stu_search = ctk.CTkEntry(
            btn_row,
            placeholder_text="Search by name or class...",
            width=240, height=32)
        self._stu_search.pack(side="left")
        self._stu_search.bind(
            "<KeyRelease>",
            lambda e: self._filter_students())

        self._stu_scroll = ctk.CTkScrollableFrame(page)
        self._stu_scroll.pack(fill="both", expand=True)

        self._all_students_data = []
        return page

    def _refresh_students(self):
        edb, _, lmap = db_load()
        self._all_students_data = [
            (sid, info, len(edb.get(sid,[])))
            for sid, info in lmap.items()
        ]
        self._render_students(self._all_students_data)

    def _filter_students(self):
        q = self._stu_search.get().strip().lower()
        filtered = [
            (sid, info, n)
            for sid, info, n in self._all_students_data
            if q in info["name"].lower()
            or q in info["class"].lower()
        ]
        self._render_students(filtered)

    def _render_students(self, data):
        for w in self._stu_scroll.winfo_children():
            w.destroy()

        # Header
        hdr = ctk.CTkFrame(
            self._stu_scroll, fg_color="#1a1d36",
            corner_radius=8)
        hdr.pack(fill="x", pady=(0,4))
        for col, w in [
            ("Name",200), ("Class",100),
            ("Embeddings",130), ("Student ID",220),
            ("Actions",130)
        ]:
            ctk.CTkLabel(
                hdr, text=col, width=w,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=ACCENT
            ).pack(side="left", padx=8, pady=8)

        if not data:
            ctk.CTkLabel(
                self._stu_scroll,
                text="No students registered yet. Go to Register.",
                text_color="gray",
                font=ctk.CTkFont(size=12)
            ).pack(pady=30)
            return

        for sid, info, n_embs in data:
            row = ctk.CTkFrame(
                self._stu_scroll,
                fg_color=CARD, corner_radius=7,
                border_width=1, border_color=BORDER)
            row.pack(fill="x", pady=2)

            emb_col = GREEN if n_embs >= 8 else YELLOW \
                      if n_embs >= 3 else RED
            for val, w, col in [
                (info["name"],  200, None),
                (info["class"], 100, None),
                (f"{n_embs} embeddings", 130, emb_col),
                (sid, 220, "gray"),
            ]:
                kw = {"text":val,"width":w,
                      "font":ctk.CTkFont(size=10)}
                if col: kw["text_color"] = col
                ctk.CTkLabel(row,**kw).pack(
                    side="left",padx=8,pady=7)

            # Delete button
            ctk.CTkButton(
                row, text="🗑 Delete",
                width=100, height=26,
                fg_color="transparent",
                border_width=1, border_color=RED,
                text_color=RED,
                font=ctk.CTkFont(size=9),
                command=lambda s=sid, n=info["name"]:
                    self._do_delete(s,n)
            ).pack(side="left", padx=6)

    def _do_delete(self, sid, name):
        if messagebox.askyesno(
            "Delete Student",
            f"Permanently delete {name}?\n\n"
            "This removes all their photos, embeddings,\n"
            "and recognition data. Cannot be undone.",
            icon="warning"
        ):
            delete_student(sid)
            messagebox.showinfo(
                "Deleted", f"{name} removed from system.")
            self._refresh_students()

    # ── Model loader ───────────────────────────────────────────────────────────
    def _load_model_bg(self):
        def _load():
            try:
                self.after(0, lambda: self.model_lbl.configure(
                    text="⏳  Loading model...",
                    text_color="orange"))
                Embedder.get()
                self._model_ready = True
                self.after(0, lambda: self.model_lbl.configure(
                    text="✅  Model ready",
                    text_color=GREEN))
            except FileNotFoundError:
                self.after(0, lambda: self.model_lbl.configure(
                    text="❌  Run setup.py first",
                    text_color=RED))
                self.after(500, lambda: messagebox.showerror(
                    "Model Not Found",
                    "ArcFace model is missing.\n\n"
                    "Run this command first:\n"
                    "  python setup.py\n\n"
                    "Then restart the app."))
            except Exception as e:
                self.after(0, lambda: self.model_lbl.configure(
                    text="❌  Model error",
                    text_color=RED))
                self.after(500, lambda: messagebox.showerror(
                    "Model Error", str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def _on_close(self):
        if self._cam_running:
            self._stop_cam()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()