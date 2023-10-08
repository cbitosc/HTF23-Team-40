import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sqlite3 as sql

ps = PorterStemmer()

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Connect to SQLite database
the_connection = sql.connect('FakeNewsDetected.db')
the_cursor = the_connection.cursor()

# Create the "news" table if it doesn't exist
the_cursor.execute('''
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        prediction INTEGER
    )
''')
the_connection.commit()

st.title("Fake News Detection App")

input_article = st.text_area("Enter the Article")

if st.button('Predict'):
    # 1. Preprocess
    transformed_article = transform_text(input_article)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_article])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("Fake News")
    else:
        st.header("Real news")

    # Save the detected news and prediction to the database
    the_cursor.execute("INSERT INTO news (message, prediction) VALUES (?, ?)", (input_article, int(result)))
    the_connection.commit()

# Close the database connection
the_cursor.close()
the_connection.close()
