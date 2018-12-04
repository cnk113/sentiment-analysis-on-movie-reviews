# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
from joblib import dump

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def remove_punctuations(text):
    for punctuation in string.punctuation:
        if punctuation == '-':
            text = text.replace(punctuation, ' ')
        else:
            text = text.replace(punctuation, '')
    return text

def main():
    df = pd.read_csv('train.csv')
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=LemmaTokenizer(),
            min_df=2,
            ngram_range=(1,2))),
        ('clf', LinearSVC())
    ])
    clf.fit(x, y)
    
    dump(clf, 'model.joblib')

if __name__ == "__main__":
    main()
