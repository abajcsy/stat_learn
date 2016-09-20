"""
Implementation of stochastic gradient descent.
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

        # extheta = np.exp((theta.T)*X[I])
        # compute the gradient at the current location
        g = y[I]*X[I] - X[I]*1.0/(1.0+np.exp(-(theta.T)*X[I]))

        # step in the direction of the gradient
        theta2 = theta + (1.0/(i+1.0))*g

        if(np.dot(theta2-theta, theta2-theta) < epsilon):
            return theta
        else:
            theta = theta2

    # return the solution
    return theta


if __name__ == '__main__':
    Xone = np.loadtxt('data_problem2.4/Xone.dat')
    yone = np.loadtxt('data_problem2.4/yone.dat')

    Xtwo = np.loadtxt('data_problem2.4/Xtwo.dat')
    ytwo = np.loadtxt('data_problem2.4/ytwo.dat')

    (m,n) = np.shape(Xone)
    # take number of iterations to be number of examples
    numIter = 10000
    epsilon = 0.0000000000001
    stepSize = 0.01

    # data set #1
    theta_hat1 = stoch_gd(Xone, yone, numIter, stepSize, epsilon)
    e_yxtheta1 = np.exp(-np.dot(Xone,theta_hat1))
    p_one = 1.0/(1.0+e_yxtheta1)

    # data set #2
    theta_hat2 = stoch_gd(Xtwo, ytwo, numIter, stepSize, epsilon)
    e_yxtheta2 = np.exp(-np.dot(Xtwo,theta_hat2))
    p_two = 1.0/(1.0+e_yxtheta2)

    print theta_hat1
    print theta_hat2

    bins = np.linspace(0, 1, 40)

    plt.title("Histogram of probabilities based on theta_hat")
    plt.xlabel("Probability: P(y_i | x_i, theta_hat)")
    plt.ylabel("Number of samples with given probability")
    plt.hist(p_one, bins)
    plt.show()
    plt.hist(p_two, bins, facecolor='green')
    plt.show()
