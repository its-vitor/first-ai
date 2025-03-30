from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import joblib
import re

cached = True

def clean_text(text): 
    return re.sub(r"[^a-zA-Z\s]", "", text).lower()

def save_model(model: RandomForestClassifier, vectorizer: TfidfVectorizer):
    joblib.dump(model, 'cache/model.pkl')
    joblib.dump(vectorizer, 'cache/vectorizer.pkl')

def load_model():
    return joblib.load('cache/model.pkl'), joblib.load('cache/vectorizer.pkl')

try: 
    model, vectorizer = load_model()
except FileNotFoundError:
    dataset = pd.read_csv("dataset/dataset.csv")
    dataset['review'] = dataset['review'].apply(clean_text)
    
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(dataset['review'])
    y = dataset['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    if cached:
        save_model(model, vectorizer)

def predict(text: str):
    result = model.predict(
        vectorizer.transform([text])
    )
    print(result)
    return "Negative Sentiment" if result[0] == 'negative' else "Positive Sentiment"

while True:
    message = input("Enter your message ('exit' to quit): ")
    if message.lower() == 'exit':
        break
    print("Sentiment:", predict(message))