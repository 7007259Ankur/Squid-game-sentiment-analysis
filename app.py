from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained sentiment model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    data = pickle.load(model_file)
    model = data['model']
    vectorizer = data['vectorizer']  # Change from 'vocab' to 'vectorizer'

# Function to transform text to vector
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c if c.isalnum() or c.isspace() else '' for c in text])
    return text

def text_to_vector(text, vectorizer):
    # Use the vectorizer to transform text to vector
    vector = vectorizer.transform([text]).toarray()  # Transform to array directly
    return vector[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['review']
    review_clean = preprocess_text(review)
    review_vec = text_to_vector(review_clean, vectorizer)  # Use vectorizer instead of vocab
    # Predict sentiment
    sentiment = model.predict([review_vec])[0]
    sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
    return render_template('index.html', review=review, sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
