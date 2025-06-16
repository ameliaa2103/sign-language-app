import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model CNN
model = tf.keras.models.load_model("sign_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

st.set_page_config(page_title="Penerjemah Video Bahasa Isyarat", layout="centered")
st.title("📺 Pembaca Video Bahasa Isyarat")
st.write("Unggah video berisi bahasa isyarat, lalu klik tombol **🧠 Terjemahkan** untuk melihat hasil prediksi huruf.")

# Upload video
video_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov", "mpeg"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Layout: dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎞️ Video Bahasa Isyarat")
        st.video(video_path)

    if st.button("🧠 Terjemahkan"):
        cap = cv2.VideoCapture(video_path)
        output_text = ""
        pred_per_frame = []
        frame_count = 0

        st.info("🔄 Memproses video frame per frame...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                continue

            # Preprocessing
            img = cv2.resize(frame, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = model.predict(img)
            pred_label = class_names[np.argmax(pred)]
            output_text += pred_label
            pred_per_frame.append(pred_label)

        cap.release()

        with col2:
            st.subheader("🔤 Hasil Translate (Huruf):")
            st.success(output_text)

            st.subheader("📜 Translate Gerakan:")
            st.markdown("".join([f"`{char}` " for char in pred_per_frame]))
