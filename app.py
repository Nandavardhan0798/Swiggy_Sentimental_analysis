# app.py
import streamlit as st
import joblib
import re

# -------------------------------
# Load the saved model and vectorizer
# -------------------------------
model = joblib.load("swiggy_sentiment_model.pkl")
vectorizer = joblib.load("swiggy_vectorizer.pkl")

# -------------------------------
# Function to clean tweet text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)           # remove mentions
    text = re.sub(r'http\S+|www\S+', '', text) # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)       # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text)           # remove extra spaces
    return text.strip()

# -------------------------------
# Streamlit app interface
# -------------------------------
st.title("Swiggy Tweet Sentiment Analyzer")
st.write("Enter a tweet below and the app will predict its sentiment (Positive, Neutral, Negative).")

# Text input
user_input = st.text_area("Enter Tweet:", "")

# Button to predict
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        # Clean and transform input
        cleaned_input = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned_input])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == "Positive":
            st.success(f"Predicted Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"Predicted Sentiment: {prediction}")
        else:
            st.info(f"Predicted Sentiment: {prediction}")
