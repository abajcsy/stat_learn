"""
Implementation of gradient descent.
"""
import numpy as np
from numpy import *

def run_gd(X, y, numIter, stepSize):
    (h,w) = X.shape
    theta = np.zeros(w)

    # set up storage for trajectory of function values
    # trajectory = zeros(numIter + 1)
    # trajectory[0] = theta

    # begin iterations
    for i in range(numIter):
        pred = np.dot(X,theta)

        error = pred - y
        cost = np.sum(error**2)/(2*y.size)

        # compute the gradient at the current location
        g = X.T.dot(error)/y.size

        # compute the step size
        # eta = stepSize/sqrt(iter+1)

        # step in the direction of the gradient
        theta = theta - stepSize*g

        # record the trajectory
        # trajectory[iter+1] = theta

    # return the solution
    return theta

if __name__ == '__main__':
    X = genfromtxt('Xmatrix.dat')
    y = genfromtxt('yvector.dat')

    numIter = 100
    stepSize = 0.01
    theta = run_gd(X, y, numIter, stepSize)
    print theta
