import nltk
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

# nltk.download('stopwords')
# from nltk.corpus import stopwords # issue with stopwords loading

class TextPrepocesser:
    '''Preprocessed text for tokenization'''
    def __init__(self, text_data: pd.DataFrame) -> None:
        self.text = text_data['review']

    def _text_lowercase(self, text):
        '''Returns text in lower case'''
        return text.lower()
    
    def _remove_numbers(self, text):
        '''Removes numbers from text'''
        return re.sub(r'\d+', '', text)

    def _remove_punctuation(self, text):
        '''Removes punctuation'''
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _remove_whitespace(self, text):
        '''Removes any whitespace'''
        return " ".join(text.split())
    
    def _remove_stopwords(self, text):
        '''Removes stopwords'''
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word not in stop_words
        ]
        return filtered_text
    
    def text_processor_pipeline(self):
        '''Returns output of all preprocessing steps into list'''
        N = self.text.shape[0]
        output = []
        for i in range(N):
            text = self.text.iloc[i]
            lower_case = self._text_lowercase(text)
            numbers_removed = self._remove_numbers(lower_case)
            punctuation_removed = self._remove_punctuation(numbers_removed)
            whitespace_removed = self._remove_whitespace(punctuation_removed)
            # stopwords_removed = self._remove_stopwords(whitespace_removed)
            output.append(whitespace_removed.split())
        return output
    
class TextTokenizer:
    '''Tokenizes text'''
    def __init__(self, processed_text: list[str]) -> None:
        self.processed_text = processed_text
        self.vocab = set()

    def _build_vocabulary(self):
        for text in self.processed_text:
            for word in text:
                self.vocab.add(word)
        self.vocab = sorted(list(self.vocab))

path = './IMDB Dataset.csv'
df = pd.read_csv(path)
processor = TextPrepocesser(df)
output = processor.text_processor_pipeline()
tokenizer = TextTokenizer(output)
tokenizer._build_vocabulary()


