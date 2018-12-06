# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn import metrics
import string
from joblib import load

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import csv

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
    df = pd.read_csv('testset_1.csv')
    xid = df['PhraseId']
    x = df['Phrase'].apply(remove_punctuations) # Features
    
    clf = load('model.joblib')
    
    yPred = clf.predict(x)
    with open('prediction.csv', mode='w', newline='') as predFile:
        predWriter = csv.writer(predFile, delimiter=',')
        predWriter.writerow(['PhraseId', 'Sentiment'])
        for i in range(len(xid)):
            predWriter.writerow([xid[i], yPred[i]])

if __name__ == "__main__":
    main()
