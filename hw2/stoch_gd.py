"""
Implementation of gradient descent.
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def log_loss(X,y,theta):
    sum = 0
    for i in range(m):
        sum += log(1+np.exp(-y[i]*((theta.T)*X[i] + b)))
    sum = -sum

def stoch_gd(X, y, numIter, stepSize, epsilon):
    (m,n) = np.shape(X)
    theta = np.random.rand(n)

    # begin iterations
    for i in range(numIter):
        I = np.random.randint(0,m)
        XIt = np.transpose(X[I])

        # compute the gradient at the current location
        g = log(np.exp(-y[I]*((theta.T)*X[I])))
        # g = -y[I]*XIt + XIt*(exp(XIt*theta)/(1+exp(XIt*theta)))

        # step in the direction of the gradient
        theta2 = theta - stepSize/(i+1)*g

        if(np.dot(theta2-theta, theta2-theta) < epsilon):
            return theta
        else:
            theta = theta2

    # return the solution
    return theta


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

    #while i > 0.0000001:
    stepSize = i/100
    res = stoch_gd(X, y, numIter, stepSize, epsilon)
    thetaX = np.transpose(res)*X
    print thetaX.T*y
    #result[j] = np.exp(thetaX.T*y)/(1+np.exp(thetaX));
    #j += 1
    #i /= 10.0

    plt.plot(np.transpose(thetaX)*y)
    #plt.axis([0, m, 0, 1.5])
    plt.show()

    # verify what alpha it converges for
    #XtX = np.dot(np.transpose(X),X)
    #n = y.shape[0]
    #lam = np.linalg.eig(XtX)[0][0]
    #print 2*n/lam
