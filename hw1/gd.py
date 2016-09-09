"""
Implementation of gradient descent.
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def gd(X, y, numIter, stepSize, epsilon):
    (m,n) = np.shape(X)
    theta = np.random.rand(n)
    Xtrans = np.transpose(X)

    # begin iterations
    for i in range(numIter):
        pred = np.dot(X,theta)
        error = pred - y

        # compute the gradient at the current location
        g = np.dot(Xtrans,error)/y.shape[0]

        # step in the direction of the gradient
        theta2 = theta - stepSize*g

        if(np.dot(theta2-theta, theta2-theta) < epsilon):
            return 1
        else:
            theta = theta2

    # return the solution
    return 0


if __name__ == '__main__':
    X = np.loadtxt('Xmatrix.dat')
    y = np.loadtxt('yvector.dat')

    numIter = 100000
    epsilon = 0.0000000000001
    i = 1
    j = 1
    result = np.empty([1000, 1])

    while i < 60:
        stepSize = i/100.0
        res = gd(X, y, numIter, stepSize, epsilon)
        print stepSize
        print res
        result[j] = res;
        j += 1
        i += 0.1

    plt.plot(result)
    plt.axis([0, 110, 0, 1.5])
    plt.show()

    # verify what alpha it converges for
    XtX = np.dot(np.transpose(X),X)
    n = y.shape[0]
    lam = np.linalg.eig(XtX)[0][0]
    print 2*n/lam
