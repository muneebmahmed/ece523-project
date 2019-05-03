#!/usr/bin/env python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import updown
import parser
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def stock_game(prices, predictions, starting_money):
	money = starting_money
	stocks = 0
	for i in range(len(predictions)):
		if predictions[i] == 1:
			stocks += money / prices[i]
			money = 0
		elif predictions[i] == -1:
			money += stocks * prices[i]
			stocks = 0
	return stocks, money + stocks * prices[len(predictions)]


def runRnn(filename):
	# load dataset
	dataset = pd.read_csv(f"./parsed/{filename}.csv", header=0, index_col=0)
	values = dataset.values.astype('float32')
	print(f"Length of csv: {len(values)}")
	
	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	# specify the number of lag hours
	n_hours = 1
	n_features = 4
	# frame as supervised learning
	reframed = series_to_supervised(scaled, n_hours, 1)
	values = reframed.values
	
	# split into train and test sets
	trainPercent = 0.975
	pivot = int(len(values) * trainPercent)
	train = values[:pivot, :]
	test = values[pivot:, :]

	# split into input and outputs
	n_obs = n_hours * n_features
	train_X, train_y = train[:, :n_obs], train[:, -n_features]
	test_X, test_y = test[:, :n_obs], test[:, -n_features]
	
	# reshape input to be 3D: (samples, timesteps, features)
	train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
	
	model = keras.models.Sequential([
		keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])),
		keras.layers.Dense(1)
	])
	model.compile(loss='mae', optimizer='adam')
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)
	
	# make prediction
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

	# invert scaling for forecast
	inv_yhat = np.concatenate((yhat, test_X[:, -3:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	# invert scaling for actual
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = np.concatenate((test_y, test_X[:, -3:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	
	# calculate RMSE
	rmse = mean_squared_error(inv_y, inv_yhat)**0.5

	# calculate updown
	real, pred = updown.up_down(inv_y), updown.up_down(inv_yhat)
	dirPredRate = updown.compare_updown(real, pred)[1]

	plt.figure()
	plt.title(f"{filename} - RMSE: {round(rmse, 3)}, Direction Prediction Rate: {round(dirPredRate, 3)}")
	plt.plot(range(len(inv_y)), inv_y, 'b')
	plt.plot(range(len(inv_yhat)), inv_yhat, 'r')
	plt.xlabel('Time (Days)')
	plt.ylabel('Stock Price ($)')
	plt.legend(['Real', 'Prediction'])
	
	#plt.show()
	plt.savefig("./output/{}.png".format(filename), bbox_inches='tight', dpi=400)

	most_stocks, most_money = stock_game(inv_y, real, 10000)
	predicted_stocks, predicted_money = stock_game(inv_y, pred, 10000)
	with open("./output/{}.txt".format(filename), "w") as f:
		f.write("Possible stocks and money: %.2f\t$%.2f\n" % (most_stocks, most_money))
		f.write("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))
	print("Total stocks and money: %.2f\t$%.2f" % (most_stocks, most_money))
	print("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))

companyList = ['AAPL', 'BA', 'KO', 'MSFT', 'NKE']
for company in companyList:
	parser.daily_parse(company)
	runRnn(company)
