# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import string

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

def main(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=LemmaTokenizer(),
            min_df=2,
            ngram_range=(1,2))),
        ('clf', LinearSVC())
    ])
    clf.fit(xTrain, yTrain)
    
    yPred = clf.predict(xTest)
    # print("Accuracy:",metrics.accuracy_score(yTest, yPred))
    return metrics.accuracy_score(yTest, yPred)

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    x = df['Phrase'].apply(remove_punctuations) # Features
    y = df['Sentiment'] # Labels
    sum = 0
    trials = 10
    for i in range(trials):
        sum += main(x, y)
    avg = sum / trials
    print(avg)
