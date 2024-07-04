import streamlit as st
import pandas as pd

from model.func import load_model, predict, extract_features

st.title("Music Genre Classification")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open('temp_audio_file.wav', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp_audio_file.wav", format="audio/mpeg", loop=True)

    # Extract features from the uploaded audio file
    features = extract_features('temp_audio_file.wav')
    
    if features:
        # st.write("Extracted Features:")
        # st.write(features)
        
        # Load the model
        model = load_model()
        
        # Predict using the model
        prediction = predict(model, pd.DataFrame(features, index=[0]))
        
        # Display prediction
        st.write("Prediction:")
        st.write(prediction)
    else:
        st.write("Could not extract features from the audio file.")