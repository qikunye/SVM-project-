import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Importing Support Vector Classifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def get_news():
    url = 'https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly9jaGF0Z3B0LmNvbS8&guce_referrer_sig=AQAAALaOq8ZqxFy3jC2RIMthAperaWbOgs2tVyabVIvXOn0nLQGXiJNTyxXPB_mR-arXMNb_ie2jqvrNYp1QLVa6WnSbUsSoptTGZYn1wY42g1WHn-fMdAZMYORJAhRaOOjH_Dq_4JHuduDebYiPAmLenCqK7GCkuozRq2JMVe2PnThh'  # Replace with a real blockchain news site
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Scrape all the headlines or articles
    headlines = []
    for headline in soup.find_all('h2'):  # Modify the tag as per the website structure
        headlines.append(headline.text)

    return headlines

headlines = get_news()
for idx, headline in enumerate(headlines[:5]):  # Show first 5 articles
    print(f"{idx+1}. {headline}")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove stopwords and stem words
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

processed_headlines = [preprocess_text(headline) for headline in headlines]
for idx, processed_headline in enumerate(processed_headlines[:5]):
    print(f"{idx+1}. {processed_headline}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_headlines)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
sentiments = [get_sentiment(headline) for headline in headlines]
for idx, sentiment in enumerate(sentiments[:5]):
    print(f"{idx+1}. Sentiment: {sentiment}")

# Prepare the labels
labels = ['Positive', 'Negative', 'Neutral']  # Adjust according to your sentiment data

# Convert sentiment into numerical values for model training
label_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
y = [label_map[sentiment] for sentiment in sentiments]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model (Support Vector Machine)
model = SVC(kernel='linear')  # You can also try other kernels like 'rbf' or 'poly'
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
import numpy as np
unique_labels = np.unique(y_test)  # Get actual labels in test data
label_names = {1: "Positive", 0: "Neutral", -1: "Negative"}

print(classification_report(y_test, y_pred, labels=unique_labels, target_names=[label_names[l] for l in unique_labels]))

# Sentiment counts
sentiment_count = {'Positive': sentiments.count('Positive'),
                   'Negative': sentiments.count('Negative'),
                   'Neutral': sentiments.count('Neutral')}

# Plot sentiment distribution
plt.bar(sentiment_count.keys(), sentiment_count.values())
plt.title('Sentiment Distribution of Blockchain News')
plt.xlabel('Sentiment')
plt.ylabel('Number of Articles')
plt.show()