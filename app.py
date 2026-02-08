import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from audio_utils import split_audio, extract_features
from emotion_model import predict_emotion
import tempfile

st.title("🎙 Voice Emotion Detection Dashboard")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    chunks = split_audio(audio_path)

    results = []

    for chunk, start, sr in chunks:
        features = extract_features(chunk, sr)
        emotion = predict_emotion(features)

        minute = int(start // 60)
        second = int(start % 60)

        results.append({
            "Time": f"{minute}:{second:02d}",
            "Emotion": emotion
        })

    df = pd.DataFrame(results)

    st.subheader("📋 Emotion Timeline")
    st.dataframe(df)

    st.subheader("📊 Emotion Distribution")
    emotion_counts = df["Emotion"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(emotion_counts, labels=emotion_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)