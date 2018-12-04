# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import time
from nltk.corpus import stopwords

   
    
def main():
    df = pd.read_csv('train.csv')
 
    # for i in range(9):
    #    print(df['Phrase'][i+20])   
    
    stop = stopwords.words('english')
    
    # stop.remove("not")
    print(stop)
    
    # print("---")
    
    # for i in range(9):
    #    print(df["Phrase"][i+20])
        
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    
    start = time.time()
    print("Training Starts at: ", time.strftime('%X %x %Z'))
    
    clf = Pipeline([
            ('vect', CountVectorizer(stop_words = stop, ngram_range=(1, 3))), # stop_words = stop
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier())
            ])
    clf.fit(xTrain, yTrain) # Build a forest of trees from the training set (x, y).
        
    print("Time Took: ", (time.time() - start))
    
    yPred = clf.predict(xTest)
    print("Accuracy:",metrics.accuracy_score(yTest, yPred))

if __name__ == "__main__":
    main()
