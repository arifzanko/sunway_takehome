'''
This code creates a simple web-based sentiment analysis app using Streamlit. 
It allows users to input a sentence, preprocesses the text, sends it to a backend service for sentiment analysis, 
and also provides feedback functionality to add the sentence to a dataset.
'''

import streamlit as st
import requests
import re

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    print(text)
    return text


def request_sentiment_analysis(sentence: str):
    url = f'http://127.0.0.1:8000/sentiment_analysis/{sentence}'
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError, if one occurred.
    return response.json()

def add_to_datasets(sentence: str):
    url = f'http://127.0.0.1:8000/add_to_datasets/{sentence}'
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError, if one occurred.
    return response.json()

st.title('Bonus Challenge: Building a Simple LLM-powered Sentiment Analyzer ')
input_text=st.text_input("Enter your sentence here")
preprocessed_text = preprocess_text(input_text)

if preprocessed_text != '':
    sentiment = request_sentiment_analysis(preprocessed_text)
    if sentiment is not None:
        st.markdown(f"Sentiment Analysis Result: {sentiment.upper()}")
    else:
        st.markdown("Sentiment analysis failed.")

    col1, col2 = st.columns(2)
    with col1:
        good_button = st.button("👍")     
    with col2:
        bad_button = st.button("👎")

    if good_button:
        add_to_datasets(preprocessed_text)
        st.success("Thank you for your feedback!")
    elif bad_button:
        st.success("Thank you for your feedback!")

