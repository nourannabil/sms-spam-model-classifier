import streamlit as st 
import pickle 
import nltk
from nltk.stem import PorterStemmer  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

stemmer = PorterStemmer()

def transform_text(text):
    text = re.sub('[^a-zA-Z]' , ' ' , text)
    text = text.lower()
    text = word_tokenize(text)
    text = [stemmer.stem(x) for x in text if x not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text



st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter The Message")

if st.button('Predict'):
    # preprocss
    transform_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transform_sms])

    # predict 
    result = model.predict(vector_input)[0]

    # Display 
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")