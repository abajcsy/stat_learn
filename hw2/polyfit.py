'''
Implementation of polynomal fit
'''

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def polyfit_D(t, y, degree):
	n = len(y)
	print n
	
	# make a n x degree+1 matrix
	X = np.zeros((n, degree+1))
	for i in range(n):
		for j in range(degree+1):
			X[i][j] = (t[i])**j
	print X
	
	XtX = np.dot(np.transpose(X),X)
	inv_XtX = np.linalg.inv(XtX)
	Xty = np.dot(np.transpose(X),y)
	coeff = np.dot(inv_XtX, Xty)
	
	return coeff

if __name__ == "__main__":
	t = np.loadtxt('data_problem2.1\\t.dat')
	y = np.loadtxt('data_problem2.1\\y.dat')
	
	print t
	print y
	
	# fit the data with a D degree polynomial
	D = 9
	coeffs = polyfit_D(t,y,D)
	rev_coeffs = np.fliplr([coeffs])[0]
	print coeffs
	print rev_coeffs
	polyn = np.poly1d(rev_coeffs) # construct the polynomial 
	
	print polyn
	xx = linspace(-1, 1, 200)
	plt.plot(t, y, 'o', xx, polyn(xx),'-r')
	plt.legend(['Data points', str(D)+'th Degree Polynomial'])
	#plt.axis([0,1,0,1])
	plt.show()