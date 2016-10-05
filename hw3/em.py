"""
Implementation of EM
"""
import numpy as np
from numpy import *
import pylab

def em(X,steps,a_0):
    n = 190
    a_t = a_0

    for i in range(steps):
        q_t = np.array([2/(2+a_t), a_t/(2+a_t), 1, 1, 1])

        num = X[0]*q_t[1] + X[3]*q_t[4]
        denom = X[0]*q_t[1] + X[3]*q_t[4] + X[1]*q_t[2] + X[2]*q_t[3]
        a_t = num/denom

    x_hat = np.array([0.5+0.25*a_t, 0.25*(1-a_t), 0.25*(1-a_t), 0.25*a_t])

    print "alpha: " + str(a_t)
    print "x_hat: " + str(x_hat*n)

if __name__ == '__main__':
    X = np.array([125,15,10,40]) # data matrix
    steps = 10 # number of steps to do
    a_0 = 0.5
    em(X, steps, a_0)


-------------------------
CONSOLE OUTPUT:
alpha: 0.747495231858
x_hat: [ 130.50602351   11.99397649   11.99397649   35.50602351]
