# Import libraries
from transformers import pipeline
import pandas as pd

# Load the sentiment analysis pipeline from Hugging Face
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to classify sentiment of a given text
def classify_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

# Sample texts for demonstration
texts = [
    "I love this movie! It's amazing.",
    "This film was terrible and a waste of time.",
    "It was okay, not the best but not the worst."
]

# Classify the sentiment for each text
results = [classify_sentiment(text) for text in texts]

# Display the results
for text, (label, score) in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {label} (Score: {score:.2f})\n")

# Optional: If you want to save results to a CSV file
results_df = pd.DataFrame(results, columns=['Sentiment', 'Score'], index=texts)
results_df.to_csv('sentiment_results.csv')
