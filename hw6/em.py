24,1"""
Implementation of EM
"""
import numpy as np
from numpy import *
import pylab

if __name__ == '__main__':
	data = np.loadtxt('hmm-gauss.dat')
	num_iter = 10000
	d = 1000

 	theta_s = np.ones((4,1))
	theta_st = np.ones((2,2))

	mu = np.ones((4,2))
	sigma_2 = np.eye(2)

	alpha1 = np.zeros((4,1))
	alpha2 = np.zeros((4,4)) 

	for l in range(num_iter):

		# ----- RUN E-STEP -----

		(A,B) = sum_prod(data, theta_s, theta_st)
		for i in range(4):
			for s in range(d):
				alpha1[i] += A[s][i]
				for j in range(4):
					alpha2[i][j] += B[s][i][j]

		# ----- RUN M-STEP -----

		# compute mu
		denom = 0.0
		for j in range(4):
			for s in range(d):
				mu[j] = data[s]*alpha1[j]
				denom += alpha1[j]
			mu[j] /= denom

		# compute sigma^2
		for s in range(d):
			for j in range(4):
				sigma += (data[s] - mu[j]).dot(data[s] - mu[j])*alpha1[j]
		sigma_2 = np.eye(2)*(sigma/2*d)

		# compute theta_s
		denom = 0
		for j in range(4):
			denom += alpha1[j]
		for j in range(4):
			theta_s[j] = math.log(alpha1[j]/denom)

		# compute theta_st
		denom = 0
		for j in range(4):
			for k in range(4):
				denom += alpha2[j][k]
		for j in range(4):
			for k in range(4):
				theta_st[j][k] = math.log(alpha2[j][k]/denom)

