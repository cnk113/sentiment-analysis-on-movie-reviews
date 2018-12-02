# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def main():
    df = pd.read_csv('train_csv')
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    clf = RandomForestClassifier()
    clf.fit(xTrain,yTrain) # Build a forest of trees from the training set (x, y).
    yPred = clf.predict(xTest)
    print("Accuracy:",metrics.accuracy_score(yTest, yPred))

if __name__ == "__main__":
    main()
