#!/usr/bin/env python

import numpy

import tensorflow as tf
from tensorflow.keras import layers

from parser import read_csv

X, y = read_csv('./data/AAPL.csv')
percentFuture = 0.15
x_train = X[0:int(len(X) * percentFuture)]; y_train = y[0:int(len(y) * percentFuture)]
x_test = X[int(len(X) * percentFuture):]; y_test = y[int(len(y) * percentFuture):]

regressor = tf.keras.models.Sequential()
regressor.add(layers.LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[0], 1)))
regressor.add(layers.Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(layers.LSTM(units = 50, return_sequences = True))
regressor.add(layers.Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(layers.LSTM(units = 50, return_sequences = True))
regressor.add(layers.Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(layers.LSTM(units = 50))
regressor.add(layers.Dropout(0.2))
# Adding the output layer
regressor.add(layers.Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

predicted = regressor.predict(x_test)

print(predicted)

