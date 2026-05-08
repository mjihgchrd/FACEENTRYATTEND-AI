import streamlit as st
import cv2
import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import csv

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FaceEntryAttend AI", layout="wide", page_icon="🎓")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { color: #8b949e; font-size: 1rem; margin-top: 4px; }

.card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
}
.stat-box {
    background: #21262d; border: 1px solid #30363d; border-radius: 10px;
    padding: 1rem; text-align: center;
}
.stat-box .val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.stat-box .lbl { font-size: 0.8rem; color: #8b949e; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb;
    margin: 2px;
}

[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, val in [
    ('camera_on', False),
    ('recognized_faces', []),
    ('registration_samples', []),
    ('reg_name', ''),
    ('reg_complete', False),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FACES_PKL  = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL  = os.path.join(DATA_DIR, 'names.pkl')
HAAR_XML   = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_haar():
    cascade = cv2.CascadeClassifier(HAAR_XML)
    if cascade.empty():
        # Fallback: use OpenCV's built-in path
        builtin = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(builtin)
    return cascade

@st.cache_resource(show_spinner=False)
def load_knn_model():
    """Load face data and train KNN — cached so it only runs once."""
    with open(FACES_PKL, 'rb') as f:
        FACES = pickle.load(f)
    with open(NAMES_PKL, 'rb') as f:
        LABELS = pickle.load(f)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    return knn, list(set(LABELS))

def invalidate_model_cache():
    load_knn_model.clear()

def detect_and_recognize(frame, facedetect, knn, confidence_threshold=0.60):
    """
    Returns annotated RGB frame + list of (name, confidence) for detected faces.
    Improvements over original:
      • Histogram equalisation on grayscale → better accuracy in poor light
      • Confidence threshold filtering
      • Larger face crop resize (70 px instead of 50) → more detail
    """
    # FIX 1: equalise histogram for better detection in varying light
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # FIX 2: slightly looser scaleFactor; minNeighbors=4 catches more faces
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4,
                                        minSize=(60, 60))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = []

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w, :]
        # FIX 3: resize to 70×70 (more detail than 50×50)
        resized = cv2.resize(crop, (70, 70)).flatten().reshape(1, -1)

        try:
            name = knn.predict(resized)[0]
            proba = knn.predict_proba(resized)[0]
            conf = float(max(proba))

            if conf >= confidence_threshold:
                color = (50, 200, 100)
                label = f"{name}  {conf:.0%}"
                results.append((name, conf))
            else:
                color = (255, 160, 40)
                label = "Unknown"

            # Draw box + label
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame_rgb, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame_rgb, label, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        except Exception:
            pass

    return frame_rgb, results

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">FaceAttend</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI Attendance System</div>', unsafe_allow_html=True)
    st.markdown("---")
    option = st.radio("Navigate", ["📊 View Attendance",
                                    "📷 Take Attendance",
                                    "👤 Register New Face"])
    st.markdown("---")

    # DB status in sidebar
    if os.path.exists(NAMES_PKL):
        with open(NAMES_PKL, 'rb') as f:
            _names = pickle.load(f)
        st.markdown(f'<div class="badge">👥 {len(set(_names))} registered</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="badge">🗂 {len(_names)} samples</div>', unsafe_allow_html=True)
    else:
        st.warning("No face database yet")

# ══════════════════════════════════════════════════════════════════════════════
# 1. VIEW ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
if option == "📊 View Attendance":
    st.markdown('<h2 style="color:#e6edf3">📊 Attendance Records</h2>', unsafe_allow_html=True)

    col_date, col_main = st.columns([1, 3])

    with col_date:
        selected_date = st.date_input("Select Date", datetime.now())
        date_str = selected_date.strftime("%d-%m-%Y")
        file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                csv_data = f.read()
            st.download_button("📥 Download CSV", data=csv_data,
                               file_name=f"Attendance_{date_str}.csv", mime="text/csv")

    with col_main:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, on_bad_lines='skip')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="stat-box"><div class="val">{len(df)}</div>'
                            f'<div class="lbl">Total Entries</div></div>', unsafe_allow_html=True)
            with c2:
                n = df['NAME'].nunique() if 'NAME' in df.columns else 0
                st.markdown(f'<div class="stat-box"><div class="val">{n}</div>'
                            f'<div class="lbl">Unique Persons</div></div>', unsafe_allow_html=True)
            st.dataframe(df, width="stretch")
            if 'NAME' in df.columns and len(df):
                st.bar_chart(df['NAME'].value_counts())
        else:
            st.info(f"No attendance recorded for **{date_str}**")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TAKE ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════
elif option == "📷 Take Attendance":
    st.markdown('<h2 style="color:#e6edf3">📷 Take Attendance</h2>', unsafe_allow_html=True)

    if not os.path.exists(FACES_PKL) or not os.path.exists(NAMES_PKL):
        st.error("⚠️ No face database found! Please register faces first (👤 Register New Face).")
        st.stop()

    facedetect = load_haar()
    knn, registered = load_knn_model()

    col_ctrl, col_info = st.columns([2, 1])
    with col_ctrl:
        c1, c2 = st.columns(2)
        with c1:
            start_btn = st.button("▶️ Start Camera", width="stretch")
        with c2:
            stop_btn  = st.button("⏹️ Stop Camera",  width="stretch")

    if start_btn:
        st.session_state.camera_on = True
        st.session_state.recognized_faces = []
    if stop_btn:
        st.session_state.camera_on = False

    frame_placeholder = st.empty()
    info_placeholder  = st.empty()

    if st.session_state.camera_on:
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not video.isOpened():
            st.error("❌ Cannot access webcam! Check permissions.")
            st.session_state.camera_on = False
        else:
            # FIX 4: set lower resolution → much faster processing
            video.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video.set(cv2.CAP_PROP_FPS, 15)

            frame_skip = 0  # FIX 5: process every other frame only

            while st.session_state.camera_on:
                ret, frame = video.read()
                if not ret:
                    break

                frame_skip += 1
                # FIX 6: run detection only every 2nd frame → ~2× speedup
                if frame_skip % 2 == 0:
                    frame_rgb, detections = detect_and_recognize(frame, facedetect, knn)
                    for name, conf in detections:
                        already = [r for r in st.session_state.recognized_faces if r[0] == name]
                        if not already:
                            ts = datetime.now().strftime("%H:%M:%S")
                            st.session_state.recognized_faces.append([name, ts])
                    frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                # Update info every frame (cheap)
                n_detected = len(st.session_state.recognized_faces)
                info_placeholder.markdown(
                    f'<div class="card">👤 <b>{n_detected}</b> unique person(s) detected this session</div>',
                    unsafe_allow_html=True)

                time.sleep(0.03)  # ~30 fps cap

            video.release()
            cv2.destroyAllWindows()

    # Show detected list
    if st.session_state.recognized_faces:
        st.markdown('<div class="card"><b>Detected this session:</b><br>', unsafe_allow_html=True)
        for name, ts in st.session_state.recognized_faces:
            st.markdown(f'<span class="badge">✓ {name} @ {ts}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Mark attendance button
    if st.button("✅ Mark Attendance", type="primary", width="stretch"):
        if not st.session_state.recognized_faces:
            st.warning("No faces detected! Start the camera and detect faces first.")
        else:
            date_str  = datetime.now().strftime("%d-%m-%Y")
            time_str  = datetime.now().strftime("%H:%M:%S")
            file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

            existing = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for row in csv.reader(f):
                        if row:
                            existing.append(row[0])

            new_entries = [e for e in st.session_state.recognized_faces
                           if e[0] not in existing]

            if new_entries:
                with open(file_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        w.writerow(['NAME', 'TIME', 'DATE'])
                    for name, ts in new_entries:
                        w.writerow([name, ts, date_str])
                st.success(f"✅ Attendance marked for {len(new_entries)} person(s)!")
                st.session_state.recognized_faces = []
            else:
                st.info("All detected persons are already marked for today.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. REGISTER NEW FACE  ← now entirely inside Streamlit, no command prompt needed
# ══════════════════════════════════════════════════════════════════════════════
elif option == "👤 Register New Face":
    st.markdown('<h2 style="color:#e6edf3">👤 Register New Face</h2>', unsafe_allow_html=True)

    facedetect = load_haar()

    # ── Step 1: name input ──────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        name_input = st.text_input("Enter person's name", placeholder="e.g. Rahul Sharma",
                                   value=st.session_state.reg_name)
        if st.button("🚀 Start Registration Camera"):
            if name_input.strip():
                st.session_state.reg_name = name_input.strip()
                st.session_state.registration_samples = []
                st.session_state.reg_complete = False
                st.session_state.camera_on = True
            else:
                st.warning("Please enter a name first!")
        st.markdown('</div>', unsafe_allow_html=True)

    TARGET_SAMPLES = 100   # same as original
    CAPTURE_EVERY  = 10    # capture 1 sample every N frames (same logic as original)

    reg_placeholder  = st.empty()
    prog_placeholder = st.empty()
    stop_reg_btn     = st.button("⏹️ Stop Registration")

    if stop_reg_btn:
        st.session_state.camera_on = False

    if st.session_state.camera_on and st.session_state.reg_name:
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not video.isOpened():
            st.error("❌ Cannot open camera!")
            st.session_state.camera_on = False
        else:
            video.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_idx = 0

            while st.session_state.camera_on:
                ret, frame = video.read()
                if not ret:
                    break

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray  = cv2.equalizeHist(gray)
                faces = facedetect.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                n_samples = len(st.session_state.registration_samples)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (50, 200, 100), 2)

                    # Capture sample every CAPTURE_EVERY frames
                    if n_samples < TARGET_SAMPLES and frame_idx % CAPTURE_EVERY == 0:
                        crop    = frame[y:y+h, x:x+w, :]
                        resized = cv2.resize(crop, (70, 70))  # 70×70 for better accuracy
                        st.session_state.registration_samples.append(resized)
                        n_samples += 1

                    label = f"Capturing {n_samples}/{TARGET_SAMPLES}"
                    cv2.rectangle(frame_rgb, (x, y-30), (x+w, y), (50, 200, 100), -1)
                    cv2.putText(frame_rgb, label, (x+4, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                reg_placeholder.image(frame_rgb, channels="RGB", width="stretch")
                prog_placeholder.progress(min(n_samples / TARGET_SAMPLES, 1.0),
                                          text=f"Collecting samples: {n_samples}/{TARGET_SAMPLES}")

                frame_idx += 1
                time.sleep(0.04)

                # Auto-stop when enough samples collected
                if n_samples >= TARGET_SAMPLES:
                    st.session_state.camera_on = False
                    st.session_state.reg_complete = True
                    break

            video.release()
            cv2.destroyAllWindows()

    # ── Save samples when done ──────────────────────────────────────────────
    if (st.session_state.reg_complete or
            len(st.session_state.registration_samples) >= TARGET_SAMPLES):

        samples = st.session_state.registration_samples[:TARGET_SAMPLES]
        name    = st.session_state.reg_name

        if samples and name:
            faces_arr = np.asarray(samples).reshape(len(samples), -1)  # flatten

            # --- Update names.pkl ---
            if os.path.exists(NAMES_PKL):
                with open(NAMES_PKL, 'rb') as f:
                    existing_names = pickle.load(f)
            else:
                existing_names = []
            existing_names.extend([name] * len(samples))
            with open(NAMES_PKL, 'wb') as f:
                pickle.dump(existing_names, f)

            # --- Update faces_data.pkl ---
            if os.path.exists(FACES_PKL):
                with open(FACES_PKL, 'rb') as f:
                    existing_faces = pickle.load(f)
                faces_arr = np.append(existing_faces, faces_arr, axis=0)
            with open(FACES_PKL, 'wb') as f:
                pickle.dump(faces_arr, f)

            # Clear KNN cache so next attendance session uses updated model
            invalidate_model_cache()

            st.success(f"🎉 **{name}** registered successfully with {len(samples)} samples!")

            # Reset state
            st.session_state.registration_samples = []
            st.session_state.reg_name    = ''
            st.session_state.reg_complete = False

    # ── Current database ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Registered Persons")
    if os.path.exists(NAMES_PKL):
        with open(NAMES_PKL, 'rb') as f:
            all_names = pickle.load(f)
        unique = sorted(set(all_names))
        cols = st.columns(min(len(unique), 4))
        for i, n in enumerate(unique):
            count = all_names.count(n)
            with cols[i % len(cols)]:
                st.markdown(f'<div class="stat-box"><div class="val">{count}</div>'
                            f'<div class="lbl">{n}</div></div>', unsafe_allow_html=True)
    else:
        st.info("No one registered yet.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("FaceEntryAttend AI  •  Final Year Project")