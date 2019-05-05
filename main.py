#!/usr/bin/env python
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import updown
import parser
 
# convert series to supervised learning
def prepareData(data, n_in=1, n_out=1):
	"""
	Given data is processed and bundled along the time axis according to delay amounts n_in and n_out

	Args:
		data: data to prepare and split
		n_in: Amount of lag days in
		n_out: Amount of lag days out
	
	Returns:
		List of data with same length as input data but prepared for LSTM
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# Input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-1 * i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	
	reframed = pd.concat(cols, axis=1)
	reframed.columns = names
	reframed.dropna(inplace=True)

	return reframed

def stock_game(prices, predictions, starting_money):
	"""
	Hypothetical investment scenario given a starting amount and future predictions

	Args:
		prices: Simulation prices
		predictions: Binary array of success and failures of prediction
		starting_money: Amount of money we start the simulation with
	
	Returns:
		The amount of stocks at the end of the simulation
		The total amount of money left at the end of the simulation
	"""
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
	"""
	Trains and runs a LSTM RNN to predict future stock values, outputting financial analysis and graphs

	Args:
		filename: name of dataset to used located in the ./parse/ directory

	Returns:
		Nothing; outputs files to the ./outputs/ directory
	"""
	# Load dataset
	dataset = pd.read_csv(f"./parsed/{filename}.csv", header=0, index_col=0)
	values = dataset.values.astype('float32')
	print(f"Length of csv: {len(values)}")
	
	# Normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	# Preparing data for LSTM layer input
	nDays = 1; nFeatures = 4
	reframed = prepareData(scaled, nDays, nDays)
	values = reframed.values
	
	# Split into training and testing sets
	trainPercent = 0.975
	pivot = int(len(values) * trainPercent)
	train = values[:pivot, :]
	test = values[pivot:, :]

	# Split training and testing sets into input and outputs for fitting
	train_X, train_y = train[:, :nDays * nFeatures], train[:, -1 * nFeatures]
	test_X, test_y = test[:, :nDays * nFeatures], test[:, -1 * nFeatures]
	
	# Reshape to [samples x time steps x features]
	train_X = train_X.reshape((train_X.shape[0], nDays, nFeatures))
	test_X = test_X.reshape((test_X.shape[0], nDays, nFeatures))
	
	model = keras.models.Sequential([
		keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])),
		keras.layers.Dense(1)
	])
	model.compile(loss='mae', optimizer='adam')
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)
	
	# Make future predictions
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], nDays * nFeatures))

	# Invert data scale
	inv_yhat = np.concatenate((yhat, test_X[:, -3:]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]

	test_y = test_y.reshape((len(test_y), 1))
	inv_y = np.concatenate((test_y, test_X[:, -3:]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]
	
	# Calculate RMSE
	rmse = mean_squared_error(inv_y, inv_yhat)**0.5

	# Calculate binary increase to decrease ratio
	real, pred = updown.up_down(inv_y), updown.up_down(inv_yhat)
	dirPredRate = updown.compare_updown(real, pred)[1]

	# Generate and save graph
	plt.figure()
	plt.title(f"{filename} - RMSE: {round(rmse, 3)}, Direction Prediction Rate: {round(dirPredRate, 3)}")
	plt.plot(range(len(inv_y)), inv_y, 'b')
	plt.plot(range(len(inv_yhat)), inv_yhat, 'r')

	plt.xlabel('Time (Days)') if sys.argv[1] == 'daily' else plt.xlabel('Time (Weeks)')
	plt.ylabel('Stock Price ($)')
	plt.legend(['Real', 'Prediction'])
	
	# plt.show()
	plt.savefig("./output/{}.png".format(filename), bbox_inches='tight', dpi=400)

	# Run hypothetical investment scenarios and write results to file
	most_stocks, most_money = stock_game(inv_y, real, 10000)
	predicted_stocks, predicted_money = stock_game(inv_y, pred, 10000)
	with open("./output/{}.txt".format(filename), "w") as f:
		f.write("Possible stocks and money: %.2f\t$%.2f\n" % (most_stocks, most_money))
		f.write("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))
	print("Total stocks and money: %.2f\t$%.2f" % (most_stocks, most_money))
	print("Predicted stocks and money: %.2f\t$%.2f" % (predicted_stocks, predicted_money))


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Please specify the model mode, either 'daily' or 'weekly'")
		quit()

	companyList = ['AAPL', 'BA', 'KO', 'MSFT', 'NKE']

	for company in companyList:
		if sys.argv[1] == 'daily':
			parser.daily_parse(company)
		elif sys.argv[1] == 'weekly':
			parser.weekly_parse(company)
		else:
			print("Undefined mode")
			quit()
		runRnn(company)
