"""
Implementation of 2-component GMM.
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn import mixture

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

def get_mean_cov(X):
    m = np.mean(X, axis=0)
    cov = np.cov(np.transpose(X))
    return (m, cov)

if __name__ == '__main__':
    Xone = np.loadtxt('data_problem2.4/Xone.dat')
    Xtwo = np.loadtxt('data_problem2.4/Xtwo.dat')
    yone = np.loadtxt('data_problem2.4/yone.dat')
    ytwo = np.loadtxt('data_problem2.4/ytwo.dat')

    (Xone1, Xone0) = parse_data(Xone, yone)
    (Xtwo1, Xtwo0) = parse_data(Xtwo, ytwo)

    print "Xone results"
    (m1, c1) = get_mean_cov(Xone1)
    (m0, c0) = get_mean_cov(Xone0)
    print m1
    print c1

    print m0
    print c0

    print "Xtwo results"
    (m21, c21) = get_mean_cov(Xtwo1)
    (m20, c20) = get_mean_cov(Xtwo0)
    print m21
    print c21

    print m20
    print c20


    #X = np.concatenate((Xone, Xtwo), axis=1)
    #run_gmm(X, 2)
