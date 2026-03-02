import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- NLTK Setup for Cloud Deployment ---
# This ensures the app downloads the required data on the Streamlit server
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Tokenize into words
    text = nltk.word_tokenize(text)

    # 3. Keep only alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # 4. Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # 5. Apply Porter Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Load the Pickle Files ---
# Ensure vectorizer.pkl and model.pkl are in the same GitHub folder
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="📧")

st.title("Email/SMS Spam Classifier")
st.markdown("Enter a message below to check if it's **Spam** or **Ham (Not Spam)**.")

input_sms = st.text_area("Enter the message", height=150)

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display Result
        if result == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM (Ham)")

# --- Footer ---
st.sidebar.info("This model uses TF-IDF Vectorization and a Multinomial Naive Bayes Classifier.")

