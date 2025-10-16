import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the small sample dataset
df = pd.read_csv('twitter_sentiment_sample.csv')

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['airline_sentiment']

# Create a simple classifier
model = LogisticRegression(max_iter=1000)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=2)  # Use cv=2 since dataset is tiny

# Print results
print("Cross-Validation Scores:", scores)
print("Mean Accuracy: {:.2f}%".format(scores.mean() * 100))


