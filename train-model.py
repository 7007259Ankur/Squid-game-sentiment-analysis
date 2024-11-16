import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load IMDb dataset
df = pd.read_csv('IMDB.csv')

# Preprocessing
df = df[['review', 'sentiment']].dropna()
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c if c.isalnum() or c.isspace() else '' for c in text])
    return text

# Apply preprocessing to the reviews
df['review'] = df['review'].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Use CountVectorizer to convert text to features
vectorizer = CountVectorizer(max_features=1000)  # Limit the vocabulary to top 10,000 words
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate the model
accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save the model and vectorizer (vocabulary)
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
print("Model and vectorizer saved!")
