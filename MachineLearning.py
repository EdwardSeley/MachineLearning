import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style 

def batch_gradient_descent(X, y):
	numOfSamples = len(X)
	numOfFeatures = X.shape[1] 
	theta = np.matrix(np.zeros(numOfFeatures)).T
	alpha = 0.00001
	print(computeCost(X, y, theta))
	newTheta = theta
	for iteration in range(1000):
		errors = X.dot(theta) - y
		for featureNum in range(numOfFeatures):
			partialDerivative = np.multiply(errors, X[:, featureNum])
			newTheta[featureNum, 0] = theta[featureNum, 0] - alpha * 1/numOfSamples * partialDerivative.sum()
		theta = newTheta
	print(computeCost(X, y, theta))

	return newTheta

def compute_cost(X, y, theta):
	cost = y - X.dot(theta)
	return np.power(cost.sum(), 2)

def computeCost(X, y, theta):  
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner) / (2 * len(X))

def run():
	data = pd.read_csv("data_file.csv", sep=",", header=0)
	X = data.ix[:, 0]
	xArray = np.asarray(X)
	X = np.column_stack((xArray, np.ones(len(X))))
	X = np.matrix(X)
	y = data.ix[:, 1]
	yArray = np.asarray(y)
	y = np.matrix(y)
	theta = batch_gradient_descent(X, y)
	plt.scatter(xArray, yArray)
	x = np.arange(xArray.min(), xArray.max(), 1)
	f = theta[0, 0] + x * theta[1, 0] 
	plt.plot(x, f, 'r')
	plt.show()
	#trainingExample = np.array(points[0])
	#parameters = linear_regression(points, [None] * points.shape[1])
	#xValues = points[:, 0]
	#yValues = points[:, 1]
	#style.use('ggplot')
	#plt.plot(xValues, yValues, 'r')
	#plt.plot(xValues, parameters[0] + xValues * parameters[1], 'b')
	#plt.show()


if __name__ == '__main__':
	run()