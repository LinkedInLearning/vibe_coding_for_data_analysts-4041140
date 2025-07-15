###################################
###   Deploying your model      ###
### in a streamlit application. ###
###################################

import joblib
import streamlit as st
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from xgboost import XGBClassifier

st.title("Song Popularity Predictor")

# User input for song features
lyrics = st.text_area("Enter Song Lyrics", "Input the lyrics of the song here...")
genre = st.selectbox("Select Genre", ['Alternative', 'Christian', 'Country', 'Hip-Hop', 'Pop', 'Rock'])
songwriter_count = st.number_input("Enter Songwriter Count", min_value=1)
producer_count = st.number_input("Enter Producer Count", min_value=1)

model = joblib.load("xgb_model.joblib")

profanity_list = ["damn", "hell", "shit", "fuck", "bitch"]

genre_mapping = pd.DataFrame({
    "Alternative": [0],
    "Christian": [0],
    "Country": [0],
    "Hip-Hop": [0],
    "Pop": [0],
    "Rock": [0]
})

genre_mapping.loc[0, genre] = 1

def preprocess_lyrics(lyrics):
    # Compute text statistics
    sentiment = SentimentIntensityAnalyzer().polarity_scores(lyrics)

    # Create a DataFrame with the features
    features = pd.DataFrame({
        "sentiment": [sentiment["compound"]],
        "word_count": [len(lyrics.split())],
        "profanity_count": [sum(1 for word in lyrics.split() if word.lower() in profanity_list)],
        "producer_count": [producer_count],
        "songwriter_count": [songwriter_count],
        "smog_index": [textstat.smog_index(lyrics)],
        "difficult_words": [textstat.difficult_words(lyrics)],
        "profanity_proportion": [sum(1 for word in lyrics.split() if word.lower() in profanity_list) / len(lyrics.split())]
        })
    features = pd.concat([features, genre_mapping], axis=1)
    features = features.fillna(0)  # Fill NaN values with 0
    return features

if st.button("Predict"):
    # Process input and generate prediction
    features = preprocess_lyrics(lyrics)
    features = features.values  # Convert DataFrame to numpy array for prediction
    features = features.reshape(1, -1)  # Reshape for single prediction
    prediction = model.predict(features)
    st.write(f"Predicted Popularity: {prediction}")