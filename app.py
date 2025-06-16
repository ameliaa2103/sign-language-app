import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="Penerjemah Bahasa Isyarat", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .title-style {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3F72AF;
        }
        .info-box {
            background-color: #DBE2EF;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# HEADER & PENGANTAR
# ==========================
st.markdown('<h1 class="title-style">🤖 Pembaca Video Bahasa Isyarat</h1>', unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>Unggah video berisi <strong>Bahasa Isyarat Indonesia</strong> (format MP4, AVI, MOV, MPEG), "
    "lalu klik tombol <strong>Terjemahkan</strong> untuk mendapatkan hasil huruf-huruf dari isyarat yang terdeteksi.</div>",
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL & KELAS
# ==========================
try:
    model = tf.keras.models.load_model("sign_model.h5")
    class_names = [chr(i) for i in range(97, 123)]  # huruf a-z
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan file `sign_model.h5` tersedia.")
    st.stop()

# ==========================
# UPLOAD VIDEO
# ==========================
video_file = st.file_uploader("📤 Upload Video Bahasa Isyarat", type=["mp4", "avi", "mov", "mpeg"])

# ==========================
# HASIL TERJEMAHAN
# ==========================
st.markdown("### 📄 Hasil Terjemahan:")

if video_file is not None:
    # Tampilkan video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    st.video(video_path)

    if st.button("🧠 Terjemahkan"):
        cap = cv2.VideoCapture(video_path)
        output_text = ""
        frame_count = 0

        if not cap.isOpened():
            st.error("❌ Tidak dapat membaca video.")
        else:
            with st.spinner("🔍 Mendeteksi bahasa isyarat dari video..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 10 != 0:
                        continue

                    # Preprocessing frame
                    img = cv2.resize(frame, (64, 64))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Prediksi
                    pred = model.predict(img, verbose=0)
                    pred_label = class_names[np.argmax(pred)]
                    output_text += pred_label

                cap.release()

            if output_text:
                st.success("✅ Terjemahan selesai!")
                st.write("🔡 **Huruf per huruf**:")
                st.code(" + ".join(list(output_text)), language="text")

                st.write("🔤 **Gabungan Kata:**")
                st.info(f"**{output_text}**")
            else:
                st.warning("⚠️ Tidak ada huruf terdeteksi dari video.")

else:
    st.info("Silakan unggah file video terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("Made with ❤️ by *Your Name* — 2025")
