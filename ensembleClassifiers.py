# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

def main():
    df = pd.read_csv('train.csv')
    x = df['Phrase'] # Features
    y = df['Sentiment'] # Labels
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3) # 80% training and 20% test
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(df['Phrase'].values)
    X = tokenizer.texts_to_sequences(df['Phrase'].values)
    X = pad_sequences(X, maxlen=2000)
    embed_dim = 128
    lstm_out = 196
    num_words = 2000
    ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(9,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
    print(model.summary())
    batch_size = 32 
    model.fit(xTrain, yTrain, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])
    model = load_model('keras_model')
    probas = model.predict(X_test)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.array(range(1, 10))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))

if __name__ == "__main__":
    main()
