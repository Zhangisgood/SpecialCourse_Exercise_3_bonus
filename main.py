import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

with open('mobydick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

sid = SentimentIntensityAnalyzer()

sentiment_scores = sid.polarity_scores(text)

average_score = (sentiment_scores['pos'] - sentiment_scores['neg'])

print(f"Average sentiment score: {average_score}")

if average_score > 0.05:
    print("The text sentiment is positive.")
else:
    print("The text sentiment is negative.")
