from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split

import utils as model_utils

def model(window_size, n_features):

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Activation('relu'))
    model.add(LSTM(100, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

def test_model():
    X, Y = model_utils.get_dummy_X_n_Y()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # Create a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # Create an Adam optimizer with the learning rate schedule
    opt = Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False)

    batch_size = int(X_train.shape[0]/10)

    # Build the LSTM model
    model_ = model(X.shape[1], X.shape[2])

    # Compile the model
    model_.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model_.fit(X_train,Y_train,epochs=2,batch_size=batch_size,verbose=1)

    model_.save('./saved_models/lstm_model_test.keras')

if __name__ == '__main__':
    test_model()