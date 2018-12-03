# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  

def main():
    df = pd.read_csv('train.csv')
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    clf.fit(xTrain, yTrain) # Build a forest of trees from the training set (x, y).
    
    yPred = clf.predict(xTest)
    print("Accuracy:",metrics.accuracy_score(yTest, yPred))

if __name__ == "__main__":
    main()
