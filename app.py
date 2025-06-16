import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf

# Konfigurasi Streamlit
st.set_page_config(page_title="Penerjemah Bahasa Isyarat", layout="wide")
st.title("📹 Pembaca Video Bahasa Isyarat")

# Load model CNN
model = tf.keras.models.load_model('sign_model.h5')
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # asumsi model output-nya 26 kelas

# Upload video
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mpeg"])

if uploaded_file is not None:
    # Simpan ke file temporer
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Layout 2 kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Video Asli")
        st.video(tfile.name)

    with col2:
        st.subheader("🔠 Hasil Translate Huruf")

        # Tombol untuk mulai menerjemahkan video
        if st.button("🔤 Terjemahkan Video"):
            cap = cv2.VideoCapture(tfile.name)
            pred_text = ""
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Hanya proses setiap 10 frame agar cepat
                if frame_count % 10 == 0:
                    img = cv2.resize(frame, (64, 64))  # Sesuaikan input model
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)

                    prediction = model.predict(img, verbose=0)
                    class_index = np.argmax(prediction)
                    predicted_letter = class_names[class_index]
                    pred_text += predicted_letter + " "

                frame_count += 1

            cap.release()

            st.success("✅ Video berhasil diterjemahkan!")
            st.markdown(f"### 🧠 Hasil Prediksi:\n**{pred_text.strip()}**")
