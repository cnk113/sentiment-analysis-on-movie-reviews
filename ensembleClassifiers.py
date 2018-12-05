# Chang Kim, Bryan Tor, Carson Burr

import pandas as pd
import numpy as np
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
    tokenizer = Tokenizer(num_words=400)
    tokenizer.fit_on_texts(df['Phrase'].values)
    X = tokenizer.texts_to_sequences(df['Phrase'].values)
    max_words = 400
    embed_dim = 128
    lstm_out = 196
    X = pad_sequences(X, max_words)
    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['acc'])
    print(model.summary())
    batch_size = 32
    model.fit(X, pd.get_dummies(y).values, epochs=8, batch_size=batch_size, validation_split=0.2)
    model.save('model')
    model = load_model('model')
    df2 = pd.read_csv('testset_1.csv')
    xTest = df2['Phrase']
    yTest = df2['Sentiment']
    probas = model.predict(xTest)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.array(range(5))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(classes[np.argmax(yTest, axis=1)], probas)))
    print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(yTest, axis=1)], preds)))

if __name__ == "__main__":
    main()
