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


if __name__ == '__main__':
    Xone = np.loadtxt('data_problem2.4/Xone.dat')
    Xtwo = np.loadtxt('data_problem2.4/Xtwo.dat')

    yone = np.loadtxt('data_problem2.4/yone.dat')
    ytwo = np.loadtxt('data_problem2.4/ytwo.dat')

    X = np.concatenate((Xone, Xtwo), axis=1)

    run_gmm(X, 2)
