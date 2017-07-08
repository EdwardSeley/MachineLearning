import numpy as np

#def batch_gradient_descent(points):

def run():
	points = np.genfromtxt('AirPassengers.csv', delimiter=',', skip_header = 1);
	print(points[][1])

if __name__ == '__main__':
	run()