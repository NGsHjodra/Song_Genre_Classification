import streamlit as st
import joblib
import librosa
import numpy as np
import os

@st.cache_data
def load_model():
    path = os.path.join('model', 'random_forest_model.pkl')
    # path = os.path.join('model', 'xgb_model.pkl') forgot to map the label to the genre
    return joblib.load(path)

def predict(model, features):
    predictions = model.predict(features)
    return predictions

def extract_features(file_name):
    # Load audio file
    try:
        # current working directory
        cwd = os.getcwd()
        print(cwd, file_name)

        y, sr = librosa.load(file_name, duration=30)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        # Aggregate features
        features = {
            'mfccs_mean': np.mean(mfccs, axis=1),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate)
        }

        # Flatten the dictionary
        flattened_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    flattened_features[f'{key}_{i}'] = v
            else:
                flattened_features[key] = value

        return flattened_features
    except:
        return None