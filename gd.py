"""
Implementation of gradient descent.
"""
import numpy as np
from numpy import *

def run_gd(X, y, numIter, stepSize):
    (h,w) = X.shape
    theta = np.zeros(w)

    # begin iterations
    for i in range(numIter):
        pred = np.dot(X,theta)

        error = pred - y

        # compute the gradient at the current location
        g = np.dot(np.transpose(X),error)/y.size
        #g = X.T.dot(error)/y.size

        # step in the direction of the gradient
        theta = theta - stepSize*g

    # return the solution
    return theta


if __name__ == '__main__':
    X = genfromtxt('Xmatrix.dat')
    y = genfromtxt('yvector.dat')

    #X = np.loadtxt('Xmatrix.dat')
    #y = np.loadtxt('yvector.dat')

    print X
    print y

    numIter = 10000
    stepSize = 0.0
    while stepSize <= 1.0:
        theta = run_gd(X, y, numIter, stepSize)
        print stepSize
        print np.linalg.norm(theta)
        stepSize += 0.05
