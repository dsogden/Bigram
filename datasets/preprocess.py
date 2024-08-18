import nltk
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

# nltk.download('stopwords')
# from nltk.corpus import stopwords

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
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word not in stop_words
        ]
        return filtered_text
    
    def text_processor_pipeline(self):
        N = self.text.shape[0]
        output = []
        for i in range(N):
            text = self.text.iloc[i]
            lower_case = self._text_lowercase(text)
            numbers_removed = self._remove_numbers(lower_case)
            punctuation_removed = self._remove_punctuation(numbers_removed)
            whitespace_removed = self._remove_whitespace(punctuation_removed)
            # stopwords_removed = self._remove_stopwords(whitespace_removed)
            output.append(whitespace_removed)
        return output

path = './IMDB Dataset.csv'
df = pd.read_csv(path)
processor = TextPrepocesser(df)
output = processor.text_processor_pipeline()
print('Before preprocessing')
print(df['review'].iloc[0])
print('After')
print(output[0])

