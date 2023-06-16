import streamlit as st
import pickle
import nltk
nltk.download('wordnet')
nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps=PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def process_text (x):
    x = x.lower()
    x = nltk.word_tokenize(x)
    y = []
    for i in x:
        if i.isalnum():
            y.append(i)
    x = y[:]
    y.clear()
    for i in x :
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    x = y[:]
    y.clear()
    for i in x:
        y.append(ps.stem(i))
    return ' '.join(y)

st.title("Text Classifier")

input_text = st.text_area("Enter Text")


if st.button('Predict'):
    transformed_text = process_text(input_text)

    vector_input = tfidf.transform([transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
