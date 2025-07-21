import joblib

import streamlit as st
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from better_profanity import profanity

st.title("Text Analysis Application")

# User inputs
text = st.text_area("Enter your lyrics:")
genre = st.selectbox("Select Genre:", ['Alternative', 'Christian', 'Country', 'Hip-Hop', 'Pop', 'Rock'])
songwriter_count = st.number_input("Songwriter Count:", min_value=1, value=1)
producer_count = st.number_input("Producer Count:", min_value=1, value=1)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Textstat metrics
        difficult_words = textstat.difficult_words(text)
        smog_index = textstat.smog_index(text)
        word_count = len(text.split())

        # Sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)

        # Profanity analysis
        profanity.load_censor_words()
        words = text.split()
        profane_words = [w for w in words if profanity.contains_profanity(w)]
        profanity_count = len(profane_words)
        profanity_proportion = profanity_count / word_count if word_count > 0 else 0

        # Prepare features for model
        import numpy as np
        import pandas as pd
        # Genre one-hot encoding
        genres = ['Alternative', 'Christian', 'Country', 'Hip-Hop', 'Pop', 'Rock']
        genre_features = [1 if genre == g else 0 for g in genres]

        features = [
            sentiment['compound'],
            word_count,
            profanity_count,
            producer_count,
            songwriter_count,
            smog_index,
            difficult_words,
            profanity_proportion
        ] + genre_features

        features_array = np.array(features).reshape(1, -1)

        # Load model and predict
        try:
            model = joblib.load("../models/xgb_model.joblib")
        except FileNotFoundError:
            try:
                model = joblib.load("models/xgb_model.joblib")
            except FileNotFoundError:
                try:
                    model = joblib.load("xgb_model.joblib")
                except Exception as e:
                    st.error(f"Model file not found: {e}")
                    model = None

        prediction = None
        if model is not None:
            try:
                prediction = model.predict(features_array)[0]
            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.subheader("Results")
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Difficult Words:** {difficult_words}")
        st.write(f"**SMOG Index:** {smog_index:.2f}")
        st.write(f"**Profane Words:** {profanity_count}")
        st.write(f"**Profanity Proportion:** {profanity_proportion:.3f}")
        st.write(f"**Sentiment (compound):** {sentiment['compound']:.3f}")
        st.write(f"**Sentiment (pos/neu/neg):** {sentiment['pos']:.3f} / {sentiment['neu']:.3f} / {sentiment['neg']:.3f}")

        if prediction is not None:
            st.success(f"**Predicted Popularity:** {prediction}")

        st.markdown("---")
        st.write(f"**Genre:** {genre}")
        st.write(f"**Songwriter Count:** {songwriter_count}")
        st.write(f"**Producer Count:** {producer_count}")

