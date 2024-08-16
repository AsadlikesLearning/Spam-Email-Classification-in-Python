import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_data(file_path):
    data = pd.read_csv('/content/Labeledspam.csv')
    return data

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_data(data):
    print("Preprocessing data... This may take a while.")
    data['processed_text'] = data['email'].apply(preprocess_text)
    return data

def feature_selection(X, y):
    selector = SelectKBest(chi2, k=2000)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    time_taken = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return accuracy, time_taken, conf_matrix, class_report

def main():
    print("Loading data...")
    data = load_data('/content/limited_dataset.csv')

    preprocessed_data = preprocess_data(data)

    print("Vectorizing text data...")
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(preprocessed_data['processed_text'])
    y = preprocessed_data['label']

    print("Performing feature selection...")
    X_selected, selector = feature_selection(X, y)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split_data(X_selected, y)

    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree (J48)': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")
        accuracy, time_taken, conf_matrix, class_report = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
        results[name] = {
            'accuracy': accuracy,
            'time': time_taken,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    
    print("\nFinal Comparison:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Time: {result['time']:.4f} seconds")

    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest model: {best_model}")
    print(f"Best accuracy: {results[best_model]['accuracy']:.4f}")

if __name__ == "__main__":
    main()
