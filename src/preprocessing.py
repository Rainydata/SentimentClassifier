import re, string, nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.base import BaseEstimator, TransformerMixin

class CleanText(BaseEstimator, TransformerMixin):
    "transformer that normalizes, tokenizes, remove stop words, lemmatizes"

    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    MENT_RE = re.compile(r"@\w+")
    HASH_RE = re.compile(r"#\w+")
    HTML_RE = re.compile(r"<.*?>")

    def __init__(self, keep_negation=True):

        self.stop_words = frozenset(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.keep_negation = keep_negation

    def fit(self, X, y=None):
        return self
    
    #return an array with cleaned text
    def transform(self, X):
        return np.array([self.clean_text(text) for text in X])
    
    def clean_text(self, text):
        
        text = text.lower()
        text = self.URL_RE.sub(" ", text)
        text = self.MENT_RE.sub(" ", text)
        text = self.HASH_RE.sub(" ", text)
        text = self.HTML_RE.sub(" ", text)

        #negation_handling
        if self.keep_negation:
            text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

        #replace punctuation to a blank space 
        text = re.sub(f"[{re.escape(string.punctuation)}]"," ", text)
        #replace numbers to space
        text = re.sub(r"\d+", " ", text)
        #replace double space to a space
        text = re.sub(r"\s+", " ", text).strip()

        tokens = wordpunct_tokenize(text)

        tokens = [self.lemmatizer.lemmatize(tok)
                  for tok in tokens 
                  if tok.lower() not in self.stop_words]

        #convert a string again
        return " ".join(tokens)