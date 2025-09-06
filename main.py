import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# App title
st.title("News Classification Using Naive Bayes")


# Cache function to load model and data (use @st.cache_resource for caching)
@st.cache_resource
def load_data():
    train = load_files("D:/20news-bydate-train", categories=[
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.mac.hrdware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'sci.crypt', 'sci.electronics', 'sci.space', 'sci.med', 'soc.religion.christian',
        'rec.sport.baseball', 'rec.sport.hockey', 'talk.politics.guns', 'talk.politics.mideast'
    ], encoding='latin1')

    test = load_files("D:/20news-bydate-test", categories=[
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.mac.hrdware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'sci.crypt', 'sci.electronics', 'sci.space', 'sci.med', 'soc.religion.christian',
        'rec.sport.baseball', 'rec.sport.hockey', 'talk.politics.guns', 'talk.politics.mideast'
    ], encoding='latin1')

    return train, test


# Load dataset
train, test = load_data()

# Create a pipeline with TfidfVectorizer and Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model with the training data
model.fit(train.data, train.target)


# Function to predict category for new input
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# User input for news heading classification
st.subheader("Classify News Heading")
user_input = st.text_input("Enter a news heading:")

if user_input:
    prediction = predict_category(user_input)
    st.write("Prediction:", prediction)
