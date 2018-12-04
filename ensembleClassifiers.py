# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import time
from nltk.corpus import stopwords

# baseline without cleaning: 60.54679% 60.7969975% 60.8214078% 60.93735% 60.52543%
# baseline with cleaning:    60.63222% 60.940408% 60.85192% 60.78174% 60.766484% 60.44915%
def clean_text(text):
    import string
    text = text.lower()
    translator = str.maketrans('','', string.punctuation)
    return text.translate(translator)
    
    
def main():
    df = pd.read_csv('train.csv')
 
    # for i in range(9):
    #    print(df['Phrase'][i+20])   
    
    stop = stopwords.words('english')
    
    df['Phrase'] = df['Phrase'].apply(clean_text)
    # df["test"] = df["Phrase"].apply(nltk.word_tokenize)
    # df['Phrase'] = df['test'].apply(lambda x: [item for item in x if item not in stop])
    
    stop.remove("not")
    print(stop)
    
    # print("---")
    
    # for i in range(9):
    #    print(df["Phrase"][i+20])
        
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    
    start = time.time()
    print("Training Starts at: ", start)
    
    clf = Pipeline([
            ('vect', CountVectorizer(stop_words = stop)), # stop_words = stop
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier())
            ])
    #clf.fit(xTrain, yTrain) # Build a forest of trees from the training set (x, y).
        
    print("Time Took: ", (time.time() - start))
    
    #yPred = clf.predict(xTest)
    print("Accuracy:",metrics.accuracy_score(yTest, yPred))

if __name__ == "__main__":
    main()
