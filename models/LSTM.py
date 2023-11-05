from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

def model(window_size, n_features):

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Activation('relu'))
    model.add(LSTM(100, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model