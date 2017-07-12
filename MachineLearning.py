import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style 

def batch_gradient_descent(points, thetaVector):
	alpha = 0.01
	newTheta = [None] * len(thetaVector)
	for x in range(len(thetaVector)):
		newTheta[x] = thetaVector[x] - alpha * average_partial_derivative(points, thetaVector, x)
	thetaVector = newTheta
	return thetaVector

def average_partial_derivative(points, thetaVector, thetaIndex):
	averagePartialDeriv = 0
	numOfTrainingExamples = points.shape[0]
	for index in range(numOfTrainingExamples):
		xValue = points[index, 0]
		features = np.array([1, xValue])
		predictedValue = np.dot(features, thetaVector)
		yValue = points[index, 1]
		partialDerivative = (yValue - predictedValue) * features[thetaIndex]
		averagePartialDeriv += partialDerivative
	averagePartialDeriv *= 0.5 * (1/numOfTrainingExamples)
	print(averagePartialDeriv)
	return averagePartialDeriv

def linear_regression(points, parameters):
	numOfFeatures = 1
	if None in parameters:
		thetaVector = np.array([0] * (numOfFeatures + 1))
	else:
		thetaVector = parameters
	parameters = batch_gradient_descent(points, thetaVector)
	return parameters

def run():
	points = np.genfromtxt('AirPassengers.csv', delimiter=',', skip_header = 1);
	trainingExample = np.array(points[0])
	parameters = linear_regression(points, [None] * points.shape[1])
	xValues = points[:, 0]
	yValues = points[:, 1]
	#style.use('ggplot')
	#plt.plot(xValues, yValues, 'r')
	plt.plot(xValues, parameters[0] + xValues * parameters[1], 'b')
	plt.show()


if __name__ == '__main__':
	run()