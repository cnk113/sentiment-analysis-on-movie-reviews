# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn import metrics
import string
from joblib import load

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import csv
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
    xid = df['PhraseId']
    x = df['Phrase'].apply(remove_punctuations) # Features
    
    # load serialized python object
    clf = load('model.joblib')
    
    # predict classifications of test set
    yPred = clf.predict(x)

    # write csv
    with open('prediction.csv', mode='w', newline='') as predFile:
        predWriter = csv.writer(predFile, delimiter=',')
        # fields
        predWriter.writerow(['PhraseId', 'Sentiment'])
        # write PhraseId - Sentiment pairs
        for i in range(len(xid)):
            predWriter.writerow([xid[i], yPred[i]])

if __name__ == "__main__":
    # use train.csv if no file specified
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = 'testset_1.csv'
    main(f)
