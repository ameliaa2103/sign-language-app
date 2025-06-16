import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("sign_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

# Streamlit config
st.set_page_config(page_title="Penerjemah Video Bahasa Isyarat", layout="centered")
st.title("📺 Pembaca Video Bahasa Isyarat")
st.write("Unggah video berisi bahasa isyarat, lalu klik tombol **🧠 Terjemahkan** untuk melihat hasil prediksi huruf.")

# Upload video
video_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov", "mpeg"])

# Simpan & tampilkan video
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    st.video(video_path)

    # Inisialisasi area hasil terjemahan
    st.markdown("---")
    st.subheader("🔤 Hasil Translate:")

    # Tombol untuk mulai menerjemahkan
    if st.button("🧠 Terjemahkan"):
        cap = cv2.VideoCapture(video_path)
        output_text = ""
        pred_per_frame = []
        frame_count = 0

        st.info("🔄 Sedang memproses video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:  # ambil setiap 10 frame
                continue

            # Preprocessing
            img = cv2.resize(frame, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict huruf
            pred = model.predict(img)
            label = class_names[np.argmax(pred)]
            output_text += label
            pred_per_frame.append(label)

        cap.release()

        # Tampilkan hasil terjemahan huruf
        st.success("✅ Terjemahan selesai")
        st.markdown(f"### 📝 Kalimat: `{output_text}`")
        st.markdown("### 📜 Translate Per Gerakan:")
        st.markdown("".join([f"`{char}` " for char in pred_per_frame]))
