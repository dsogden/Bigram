import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd
import numpy as np

class TextPrepocesser:
    def __init__(self, text_data: pd.DataFrame) -> None:
        self.text = text_data['review']

    def _text_lowercase(self, text):
        return text.lower()
    
    def _remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def _remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _remove_whitespace(self, text):
        return " ".join(text.split())
    
    def _remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word not in stop_words
        ]
        return filtered_text
    

path = './IMDB Dataset.csv'
df = pd.read_csv(path)
processor = TextPrepocesser(df)
# print(processor._text_lowercase(df['review'][0]))
