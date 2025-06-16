import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model CNN
model = tf.keras.models.load_model("sign_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

# Layout utama
st.set_page_config(page_title="Penerjemah Video Bahasa Isyarat", layout="wide")
st.title("📺 Pembaca Video Bahasa Isyarat")
st.markdown("Unggah video berisi bahasa isyarat, lalu klik tombol **Terjemahkan** untuk melihat hasil.")

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mpeg"])

# Kotak terpisah untuk video & hasil translate
col1, col2 = st.columns(2)

if video_file is not None:
    # Simpan sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Tampilkan video di sebelah kiri
    with col1:
        st.video(video_path)
        proses = st.button("🧠 Terjemahkan")

    if proses:
        cap = cv2.VideoCapture(video_path)
        output_text = ""
        frame_count = 0

        with st.spinner("🔄 Memproses video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 != 0:  # hanya ambil tiap 10 frame
                    continue

                # Preprocessing
                img = cv2.resize(frame, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # Prediksi
                pred = model.predict(img, verbose=0)
                pred_label = class_names[np.argmax(pred)]
                output_text += pred_label

            cap.release()

        with col2:
            st.success("✅ Terjemahan Selesai")
            st.subheader("📝 Hasil Translate:")
            st.code(output_text, language="markdown")
