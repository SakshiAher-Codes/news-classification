import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


st.title("News Classification Using Naive Bayes")


@st.cache_resource
def load_data():
    categories = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
        'rec.autos', 'sci.crypt', 'sci.electronics', 'sci.space',
        'sci.med', 'soc.religion.christian', 'rec.sport.baseball',
        'rec.sport.hockey', 'talk.politics.guns', 'talk.politics.mideast'
    ]

    train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    return train, test


train, test = load_data()

# Create pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(train.data, train.target)

# Prediction
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

# UI
st.subheader("Classify News Heading")
user_input = st.text_input("Enter a news heading:")

if user_input:
    prediction = predict_category(user_input)
    st.write("Prediction:", prediction)
