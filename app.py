import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("sign_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

# Setup halaman
st.set_page_config(page_title="Penerjemah Video Bahasa Isyarat", layout="centered")
st.title("📺 Pembaca Video Bahasa Isyarat")
st.write("Unggah video berisi bahasa isyarat, lalu klik tombol **Terjemahkan** untuk melihat hasil huruf-hurufnya.")

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mpeg"])

# Kotak hasil langsung ditampilkan
st.markdown("### 📝 Hasil Terjemahan:")

if video_file is not None:
    # Simpan file video sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Tampilkan video
    st.video(video_path)

    if st.button("🧠 Terjemahkan"):
        cap = cv2.VideoCapture(video_path)
        output_text = ""
        frame_count = 0

        if not cap.isOpened():
            st.error("❌ Gagal membuka video.")
        else:
            st.info("⏳ Sedang memproses video...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 != 0:  # ambil setiap 10 frame
                    continue

                # Preprocessing frame
                img = cv2.resize(frame, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                # Prediksi huruf
                pred = model.predict(img, verbose=0)
                pred_label = class_names[np.argmax(pred)]
                output_text += pred_label

            cap.release()

            # Tampilkan hasil
            if output_text:
                st.success("✅ Terjemahan selesai!")
                st.write("Huruf per huruf:")
                st.code(" + ".join(list(output_text)))

                st.write("Gabungan kata:")
                st.info(output_text)
            else:
                st.warning("⚠️ Tidak ada huruf terdeteksi dari video.")
else:
    st.info("Silakan unggah video terlebih dahulu (MP4, MOV, AVI, MPEG4, max 200MB).")
