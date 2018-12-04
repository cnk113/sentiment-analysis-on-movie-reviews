# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn import metrics
import string
from joblib import load

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
    x = df['Phrase'].apply(remove_punctuations) # Features
    y = df['Sentiment'] # Labels
    
    clf = load('model.joblib')
    
    yPred = clf.predict(x)
    print("Accuracy:", metrics.accuracy_score(y, yPred))

if __name__ == "__main__":
    main()
