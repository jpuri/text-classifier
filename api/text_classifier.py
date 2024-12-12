from fastapi import FastAPI
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

data = [
('I love my country.', 'pos'),
('This is an amazing place!', 'pos'),
('I do not like the smell of this place.', 'neg'),
('I do not like this restaurant', 'neg'),
('I am tired of hearing your nonsense.', 'neg'),
("I always aspire to be like him", 'pos'),
("It's a horrible performance.", "neg")
]

model = NaiveBayesClassifier(data)

app = FastAPI()

@app.get("/classify")
async def read_root(text: str):
    return { "class": model.classify(text) }