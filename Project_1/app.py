import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils import extract_signal, process_signal, calculate_heart_rate, get_stage

st.set_page_config(page_title="ECG Analysis ML System", layout="wide")

st.title("ECG Analysis using Machine Learning")


try:
    model = pickle.load(open("ecg_model.pkl", "rb"))
    st.success("Model loaded successfully")
except:
    st.error("Model file not found! Run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded ECG Image", use_column_width=True)

    time, signal = extract_signal(image)

    if np.std(signal) < 0.01:
        st.error("Invalid or flat ECG signal detected!")
        st.stop()

    smooth_signal, peaks = process_signal(signal)

    heart_rate, rr_interval = calculate_heart_rate(peaks, time)

    if len(peaks) > 1:
        rr_std = np.std(np.diff(time[peaks]))
    else:
        rr_std = 0

    features = np.array([[
        heart_rate,
        rr_interval,
        rr_std,
        len(peaks),
        np.var(signal)
    ]])

    # -----------------------------
    # ML PREDICTION
    # -----------------------------
    if heart_rate == 0:
        prediction = "Invalid Signal"
    else:
        try:
            prediction = model.predict(features)[0]
        except:
            prediction = "Model Error"

    abnormality = "None" if prediction == "Normal" else "Detected"


    stage = get_stage(heart_rate)


    st.subheader("Medical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Heart Rate:** {heart_rate} BPM")
        st.write(f"**RR Interval:** {rr_interval} sec")

    with col2:
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Stage:** {stage}")
        st.write(f"**Abnormality:** {abnormality}")

    if len(peaks) < 2:
        st.warning("⚠ Poor signal quality detected!")

    st.subheader("ECG Signal")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time, smooth_signal, linewidth=1.5)

    if len(peaks) > 0:
        ax.scatter(time[peaks], smooth_signal[peaks], s=40)

    ax.set_title("ECG Signal with R-Peaks")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    st.pyplot(fig)


    st.subheader("Signal Data (Sample)")

    df = pd.DataFrame({
        "Time": time,
        "Amplitude": smooth_signal
    })

    st.dataframe(df.iloc[::10].head(300))


    st.download_button(
        "⬇ Download Signal CSV",
        df.to_csv(index=False),
        "ecg_signal.csv",
        "text/csv"
    )
