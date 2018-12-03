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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

def main():
    df = pd.read_csv('train.csv')
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # 70% training and 30% test
    clf1 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', ExtraTreesClassifier())
    ])
    clf2 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    clf3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MLPClassifier())
    ])
    eclf = VotingClassifier(estimators=[('rf', clf1), ('svc', clf2), ('mlp', clf3)], voting='soft')
    grid = GridSearchCV(estimator=eclf, cv=5)
    grid = grid.fit(xTrain,yTrain)
    yPred = grid.predict(xTest)
    print("Accuracy:",metrics.accuracy_score(yTest, yPred))

if __name__ == "__main__":
    main()
