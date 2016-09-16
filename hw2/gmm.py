"""
Implementation of 2-component GMM
"""
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Xone = np.loadtxt('data_problem2.4/Xone.dat')
    yone = np.loadtxt('data_problem2.4/yone.dat')

    Xtwo = np.loadtxt('data_problem2.4/Xtwo.dat')
    ytwo = np.loadtxt('data_problem2.4/ytwo.dat')
