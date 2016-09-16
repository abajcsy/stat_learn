"""
Implementation of gradient descent.
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def stoch_gd(X, y, numIter, stepSize, epsilon):
    (m,n) = np.shape(X)
    theta = np.random.rand(n)

    # begin iterations
    for t in range(numIter):
        I = np.random.randint(0,m)
        XIt = np.transpose(X[I])

        # compute the gradient at the current location
        g = -y[I]*XIt + XIt*(exp(XIt*theta)/(1+exp(XIt*theta)))

        # step in the direction of the gradient
        theta2 = theta - stepSize/(t+1)*g

        if(np.dot(theta2-theta, theta2-theta) < epsilon):
            return 1
        else:
            theta = theta2

    # return the solution
    return 0


if __name__ == '__main__':
    X = np.loadtxt('data_problem2.4/Xone.dat')
    y = np.loadtxt('data_problem2.4/yone.dat')

    (m,n) = np.shape(X)
    # take number of iterations to be number of examples
    numIter = m
    epsilon = 0.0000000000001
    i = 1
    j = 1
    result = np.empty([m, 1])

    while i > 0.0000001:
        stepSize = i/10
        res = stoch_gd(X, y, numIter, stepSize, epsilon)
        print res
        result[j] = res;
        j += 1
        i /= 10.0

    plt.plot(result)
    plt.axis([0, m, 0, 1.5])
    plt.show()

    # verify what alpha it converges for
    #XtX = np.dot(np.transpose(X),X)
    #n = y.shape[0]
    #lam = np.linalg.eig(XtX)[0][0]
    #print 2*n/lam
