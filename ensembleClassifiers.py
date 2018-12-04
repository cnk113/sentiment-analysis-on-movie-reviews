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
from nltk.corpus import stopwords
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

def main():
    df = pd.read_csv('train.csv')
    
    df['Phrase'] = df['Phrase'].apply(remove_punctuations)
    
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    
    stop = ['there', 'about', 'once', 'during', 'having', 'with', 'they', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'is', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'should', 'our', 'their', 'while', 'both', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'them', 'and', 'been', 'have', 'in', 'will', 'does', 'yourselves', 'then', 'that', 'because', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those',  'whom', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'was', 'here', 'than', 'nt']
    
    '''tf_vec = TfidfVectorizer(min_df = 2, max_df = 0.55, ngram_range=(1,2), stop_words=stop);
    tf_vec.fit(df['Phrase']);
    
    f = open("demofile.txt","w")
    for item in tf_vec.vocabulary_.items():
        f.write(str(item) + ' \n')
    # print(tf_vec.vocabulary_)'''
    
    sum = 0;
    
    for i in range(10):
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 70% training and 30% test
    
        clf = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer(), min_df = 2, ngram_range=(1,2), stop_words=None)),
                ('clf', LinearSVC())
        ])
    
        clf.fit(xTrain, yTrain) # Build a forest of trees from the training set (x, y).
    
        yPred = clf.predict(xTest)
        print("Accuracy:",metrics.accuracy_score(yTest, yPred))
        sum += metrics.accuracy_score(yTest, yPred)
        print("Average Accuracy:", sum/(i+1))

if __name__ == "__main__":
    main()
