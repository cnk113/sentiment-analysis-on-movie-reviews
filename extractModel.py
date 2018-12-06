# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
from joblib import dump

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys

# Performs Lemmatization on the data set
#  replacing terms with their base forms
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# preprocess: removes meaningless punctuation. converts '-' to ' '
def remove_punctuations(text):
    for punctuation in string.punctuation:
        if punctuation == '-':
            text = text.replace(punctuation, ' ')
        else:
            text = text.replace(punctuation, '')
    return text

def main(f):
    df = pd.read_csv(f)
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels

    # Sequentially use Tfidf for feature extraction, then
    # Linear SVM classifier
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=LemmaTokenizer(),
            min_df=2,
            ngram_range=(1,2))),
        ('clf', LinearSVC())
    ])
    clf.fit(x, y)
    
    # Create serialized classifier object to be used in testModel.py
    dump(clf, 'model.joblib')

if __name__ == "__main__":
    # use train.csv if no file specified
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = 'train.csv'
    main(f)
