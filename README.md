# sms_spam_detector

# 📧 SMS Spam Classifier
A Machine Learning web application that classifies SMS messages as **Spam** or **Ham** (Legitimate) using Natural Language Processing (NLP).

## 🚀 Live Demo
[Insert your Streamlit Cloud URL here]

## 🛠️ Features
- **Real-time Prediction:** Enter any message and get instant results.
- **NLP Pipeline:** Uses Tokenization, Stopword removal, and Porter Stemming.
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
- **Model:** Multinomial Naive Bayes.

## 📁 Project Structure
- `app2.py`: Streamlit frontend logic.
- `model.pkl`: Trained Naive Bayes model.
- `vectorizer.pkl`: Fitted TF-IDF vectorizer.
- `requirements.txt`: Environment dependencies.
