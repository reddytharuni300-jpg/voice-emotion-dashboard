import librosa
import numpy as np

def split_audio(file_path, chunk_duration=3):
    y, sr = librosa.load(file_path)
    chunk_samples = chunk_duration * sr

    chunks = []
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i+chunk_samples]
        start_time = i / sr
        chunks.append((chunk, start_time, sr))
    return chunks

def extract_features(chunk, sr):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)