import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)

# Convert labels to numbers
data['label_num'] = data['label'].map({'real': 1, 'fake': 0})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(data['clean_text'])
y = data['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input prediction
while True:
    news = input("\nEnter news text (or type exit): ")
    if news.lower() == "exit":
        break

    news_clean = clean_text(news)
    news_vec = vectorizer.transform([news_clean])
    result = model.predict(news_vec)

    if result[0] == 1:
        print("✅ Real News")
    else:
        print("🚨 Fake News")
