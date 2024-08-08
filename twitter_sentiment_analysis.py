import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time
from tensorflow.keras.models import load_model
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

lm = WordNetLemmatizer()

# Load the model from the specified path
model_path = r"Twitter.h5"

try:
    model = load_model(model_path)
    model.summary()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def newinput(comment):
    try:
        review = re.sub('[^a-zA-Z]', ' ', comment)
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        onehot1 = [one_hot(review, n=50000)]
        encoded_comments1 = pad_sequences(onehot1, maxlen=40)
        
        if model:
            output = model.predict(encoded_comments1)
            output = np.where(output > 0.5, 1, 0)
            if output == 0:
                result = 'negative'
            else:
                result = 'positive'
            return result
        else:
            return "Model not loaded."
    except Exception as e:
        st.error(f"Error processing input: {e}")
        return "Error processing input."

def main():
    # Giving a title name
    st.title('Twitter Sentiment Analysis')
    Utweet = st.text_input('Enter your comment to classify')
    result = ''
    if st.button('Click For Result😀'):
        result = newinput(Utweet)
        
    st.success(result)

if __name__ == '__main__':
    main()
