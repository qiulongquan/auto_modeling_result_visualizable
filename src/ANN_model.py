#!/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from basic_config import X, y, verbose


def ANN(optimizer='adam',
        neurons=256,
        batch_size=32,
        epochs=50,
        activation='relu',
        patience=5,
        loss='mse'):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X.shape[1], ),
                    activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor="loss",
                                   patience=patience)  # early stop patience
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=verbose)  # verbose set to 1 will show the training process
    return model
