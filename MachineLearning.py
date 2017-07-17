import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import datasets, linear_model

def batch_gradient_descent(X, y):
	numOfSamples = len(X)
	numOfFeatures = X.shape[1] 
	theta = np.matrix(np.zeros(numOfFeatures)).T
	alpha = 0.02
	costList = [None] * 1000
	newTheta = theta
	for iteration in range(1000):
		errors = y - X.dot(theta).T
		for featureNum in range(numOfFeatures):
			partialDerivatives = errors.dot(X[:, featureNum])
			newTheta[featureNum, 0] = theta[featureNum, 0] + alpha * 1/numOfSamples * partialDerivatives
		theta = newTheta
		costList[iteration] = compute_cost(X, y, theta)
	return newTheta, costList

def compute_cost(X, y, theta):
	totalCost = X.dot(theta).T - y
	squaredErrors = np.power(totalCost, 2)
	meanSquaredError = squaredErrors.sum()/len(X)
	return meanSquaredError

def run():
	data = pd.read_csv("data_file.csv", sep=",", header=None)
	points = data.values.tolist()
	#otherTheta = gradient_descent_runner(points, 0, 0, 0.01, 1000)
	#newTheta = np.matrix([otherTheta[1], otherTheta[0]]).T
	X = data.ix[:, 0]
	xArray = np.asarray(X)
	X = np.column_stack((xArray, np.ones(len(X))))
	X = np.matrix(X)
	y = data.ix[:, 1]
	yArray = np.asarray(y)
	y = np.matrix(y)
	theta, costList = batch_gradient_descent(X, y)
	print("cost: " + str(compute_cost(X, y, theta)))
	#print("cost: " + str(compute_cost(X, y, newTheta)))
	x = np.arange(xArray.min(), xArray.max(), 1)

	f = theta[1, 0] + x * theta[0, 0]
	#plt.scatter(xArray, yArray)
	#plt.plot(x, f, "r"
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("Gradient Descent")
	ax.set_xlabel("Independent Variable")
	ax.set_ylabel("Dependent Variable")
	ax.scatter(xArray, yArray)
	ax.plot(x, f, "r")
	plt.show()

if __name__ == '__main__':
	run()
