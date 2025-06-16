import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("sign_model.h5")
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Setup halaman
st.set_page_config(page_title="Sign Language Translator", layout="wide")
st.title("🤟 ASL Sign Language Translator (A–Z)")

# Start camera toggle
run = st.toggle("🎥 Start Camera")

# Layout dua kolom
col1, col2 = st.columns([3, 1])

# Inisialisasi frame
FRAME_WINDOW = col1.image([], channels="RGB")
predicted_letter_box = col2.empty()

# Inisialisasi kamera
cap = cv2.VideoCapture(0) if run else None

if run:
    st.warning("Klik toggle lagi untuk stop kamera")

# Loop
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera tidak tersedia.")
        break

    # Crop tengah & resize
    h, w, _ = frame.shape
    roi = frame[h//4:h//4*3, w//4:w//4*3]
    roi = cv2.resize(roi, (64, 64))
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = roi_gray.reshape(1, 64, 64, 1) / 255.0

    # Prediksi huruf
    pred = model.predict(roi_gray)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Tampilkan di frame
    cv2.putText(frame, f"{pred_class} ({confidence:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Stream ke app
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Tampilkan huruf besar di kanan
    with predicted_letter_box.container():
        st.markdown("## ✍️ Prediksi Huruf")
        st.markdown(f"<div style='font-size:120px; text-align:center; color:#00BFFF;'>{pred_class}</div>", unsafe_allow_html=True)

# Stop kamera
if cap:
    cap.release()
    cv2.destroyAllWindows()
