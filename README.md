# Blockchain News Sentiment Analysis

## Overview
This project scrapes financial news headlines, processes the text, performs sentiment analysis, and trains a machine learning model to classify sentiment into Positive, Negative, or Neutral categories. The sentiment distribution is also visualized in a bar chart.

## Features
- 🔄 Scrapes financial news headlines from Yahoo Finance (or a custom URL).
- 🔄 Cleans and preprocesses text using Natural Language Processing (NLP).
- 🔄 Performs sentiment analysis using VADER.
- 🔄 Vectorizes text using TF-IDF.
- 🔄 Trains a Support Vector Machine (SVM) model for sentiment classification.
- 🔄 Visualizes the sentiment distribution of headlines.

## Installation
Ensure you have Python 3.x installed, then install the required libraries:

```bash
pip install requests beautifulsoup4 nltk scikit-learn vaderSentiment matplotlib
```

## How It Works

### 1. Scrape Blockchain News
The `get_news()` function fetches headlines from Yahoo Finance using `requests` and `BeautifulSoup`.  
> *Modify the URL to target a specific blockchain news source.*

### 2. Preprocess Text
The `preprocess_text()` function tokenizes text, removes stopwords, and applies stemming using `nltk`.

### 3. Sentiment Analysis
The `get_sentiment()` function analyzes text using `vaderSentiment` and classifies it as:
- **Positive** (compound score ≥ 0.05)
- **Negative** (compound score ≤ -0.05)
- **Neutral** (otherwise)

### 4. Train Machine Learning Model
- Converts sentiment labels into numeric values (`1` for Positive, `-1` for Negative, `0` for Neutral).
- Uses TF-IDF vectorization to transform text into numerical features.
- Splits data into training and test sets.
- Trains an **SVM (Support Vector Machine) classifier**.
- Evaluates the model using a classification report.

### 5. Visualize Sentiment Distribution
Generates a **bar chart** showing the distribution of sentiments across the headlines.

## Usage
Run the script in a Python environment:

```bash
python sentiment_analysis.py
```

### Example Output

```bash
1. Bitcoin reaches new all-time high
2. Ethereum crashes due to market panic
3. Governments push for more blockchain regulations
4. New blockchain project set to revolutionize finance
5. Investors remain uncertain about cryptocurrency trends

1. Sentiment: Positive
2. Sentiment: Negative
3. Sentiment: Neutral
4. Sentiment: Positive
5. Sentiment: Neutral
```

**Classification Report:**  
```bash
              precision    recall  f1-score   support
    Negative       0.67      0.50      0.57         2
     Neutral       0.75      1.00      0.86         3
    Positive       1.00      0.67      0.80         3
```

### Example Visualization
A bar chart displaying sentiment distribution across headlines.

## Next Steps
- Improve web scraping by targeting a dedicated blockchain news website.
- Use a larger dataset for better sentiment classification accuracy.
- Experiment with different ML models like **Naive Bayes** or **Deep Learning**.

## License
This project is licensed under the **MIT License**.

