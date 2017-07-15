import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import mean_squared_error 

def batch_gradient_descent(X, y):
	numOfSamples = len(X)
	numOfFeatures = X.shape[1] 
	theta = np.matrix(np.zeros(numOfFeatures)).T
	alpha = 0.0001
	#print(compute_cost(X, y, theta))
	costList = [None] * 50
	newTheta = theta
	for iteration in range(50):
		errors = X.dot(theta).T - y
		for featureNum in range(numOfFeatures):
			partialDerivative = np.multiply(errors, X[:, featureNum])
			newTheta[featureNum, 0] = theta[featureNum, 0] - alpha * 1/numOfSamples * partialDerivative.sum()
		theta = newTheta
		costList[iteration] = compute_cost(X, y, theta)
		alpha = 0.0001
	return newTheta, costList

def gradientDescent(X, y, theta):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(1000)

    for i in range(1000):
        error = (X * theta).T - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[j,0] = theta[j, 0] - ((0.0001 / len(X)) * np.sum(term))

        theta = temp

    return theta

def compute_cost(X, y, theta):
	totalCost = X.dot(theta).T - y
	squaredErrors = np.power(totalCost, 2)
	meanSquaredError = squaredErrors.sum()/len(X)
	return meanSquaredError

def run():
	data = pd.read_csv("data_file.csv", sep=",", header=None)
	X = data.ix[:, 0]
	xArray = np.asarray(X)
	X = np.column_stack((xArray, np.ones(len(X))))
	X = np.matrix(X)
	y = data.ix[:, 1]
	yArray = np.asarray(y)
	y = np.matrix(y)
	theta, costList = batch_gradient_descent(X, y)
	print("cost: " + str(compute_cost(X, y, theta)))
	plt.scatter(xArray, yArray)
	x = np.arange(xArray.min(), xArray.max(), 1)
	f = theta[1, 0] + x * theta[0, 0]
	plt.plot(x, f, "r")
	plt.show()

if __name__ == '__main__':
	run()
