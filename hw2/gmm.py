"""
Implementation of 2-component GMM.
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import mixture
import pylab

def run_gmm(X, num_components):
    gmm = mixture.GMM(n_components=num_components, covariance_type='full')
    gmm.fit(X)
    print gmm.means_
    colors = ['r' if i==0 else 'b' for i in gmm.predict(X)]
    p = plt.gca()
    p.scatter(X[:,0], X[:,1], c=colors)
    plt.title("2-component GMM for Xone and Xtwo datasets")
    plt.show()

#returns two matrices, one with all one labels, and other with 0 labels
def parse_data(X, y):
    X_1 = []
    X_0 = []

    (m,n) = np.shape(X)
    for i in range(m):
        if y[i] == 1.0:
            X_1.append(X[i])
        else:
            X_0.append(X[i])

    return (X_1, X_0)

def perpendicular(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def plot_gmm(X1, X0, m1, m0, theta_hat):
    diff = m0 - m1
    perp = perpendicular(diff)

    diff2 = perp - theta_hat
    slope = perp[1]/perp[0]
    b = theta_hat[1] - slope*theta_hat[0]

    x1 = 6
    y1 = slope*x1 + b

    x2 = -1
    y2 = slope*x2 + b

    xplot = [x1, x2]
    yplot = [y1, y2]
    plt.plot([x[0] for x in X1], [x[1] for x in X1], 'ob', [x[0] for x in X0], [x[1] for x in X0], 'og')
    plt.plot(m1[0], m1[1], 'or', m0[0], m0[1], 'or')
    plt.plot(xplot, yplot, '-r')
    plt.show()

def get_mean_cov(X):
    m = np.mean(X, axis=0)
    cov = np.cov(np.transpose(X))
    return (m, cov)

if __name__ == '__main__':
    Xone = np.loadtxt('data_problem2.4/Xone.dat')
    Xtwo = np.loadtxt('data_problem2.4/Xtwo.dat')
    yone = np.loadtxt('data_problem2.4/yone.dat')
    ytwo = np.loadtxt('data_problem2.4/ytwo.dat')

    theta_hat1 = [0.03971547, -1.69052142]
    theta_hat2 = [-0.01176046, -0.02374169]

    (Xone1, Xone0) = parse_data(Xone, yone)
    (Xtwo1, Xtwo0) = parse_data(Xtwo, ytwo)

    print "Xone results"
    (m1, c1) = get_mean_cov(Xone1)
    (m0, c0) = get_mean_cov(Xone0)

    plot_gmm(Xone1, Xone0, m1, m0, theta_hat1)

    print "Xtwo results"
    (m21, c21) = get_mean_cov(Xtwo1)
    (m20, c20) = get_mean_cov(Xtwo0)

    plot_gmm(Xtwo1, Xtwo0, m21, m20, theta_hat2)
