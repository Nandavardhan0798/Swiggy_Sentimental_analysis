Swiggy Twitter Sentiment Analysis
Project Overview

This project analyzes customer feedback on Twitter about Swiggy, an Indian food delivery platform. The goal is to classify tweets as Positive, Negative, or Neutral using machine learning. The workflow includes data preprocessing, handling class imbalance, feature extraction, model training, evaluation, and saving the model for future predictions.

Dataset

Source: Collected tweets mentioning @swiggy_in and @SwiggyCares.

Key columns:

date → Timestamp of tweet

full_text → Tweet content

favorite_count → Number of likes

retweet_count → Number of retweets

screen_name → Twitter handle of the user

Sentiment → Label (Neutral, Negative, Positive)

Dataset distribution (example):

Neutral     13,339
Negative    2,567
Positive    806

Features & Preprocessing

Removed mentions (@user), URLs, special characters, and numbers.

Converted text to lowercase.

Removed extra whitespace.

Balanced dataset using oversampling of minority classes (Negative & Positive) to match Neutral tweets.

Feature Extraction

TF-IDF vectorization (max 5000 features, unigrams & bigrams).

Removed English stopwords.

Model

Classifier: Multinomial Naive Bayes

Input: TF-IDF vectors of tweet text

Output: Sentiment label (Positive / Negative / Neutral)

Workflow

Load dataset.

Balance classes (oversample Negative & Positive).

Preprocess text.

Vectorize text with TF-IDF.

Split into training and testing sets (80/20).

Train the Naive Bayes classifier.

Evaluate using Accuracy, Confusion Matrix, Classification Report.

Save trained model and vectorizer (.pkl files).

Predict new tweets using saved model.

Evaluation Metrics

Accuracy: Measures overall correctness.

Confusion Matrix: Shows true vs predicted labels.

Classification Report: Includes precision, recall, and F1-score.

How to Use

Clone this repository.

Install dependencies:

pip install pandas scikit-learn matplotlib joblib


Run sentiment_analysis.py to train and evaluate the model.

Load the saved model and vectorizer to predict new tweets:

import joblib

model = joblib.load("swiggy_sentiment_model.pkl")
vectorizer = joblib.load("swiggy_vectorizer.pkl")

# Example
new_tweets = ["@swiggy_in My order is late!", "Food arrived on time, great service!"]
new_features = vectorizer.transform([clean_text(t) for t in new_tweets])
predictions = model.predict(new_features)

Insights

Most tweets are Neutral.

Negative tweets outnumber Positive → potential service issues.

Common complaints: late delivery, wrong orders, refund issues.

Positive tweets highlight fast delivery and good service.

Future Improvements

Experiment with Logistic Regression, Random Forest, or XGBoost for better accuracy.

Incorporate emoji and slang handling for improved social media sentiment detection.

Automate daily sentiment dashboard for real-time monitoring.