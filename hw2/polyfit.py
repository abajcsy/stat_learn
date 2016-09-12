'''
Implementation of polynomal fit
'''

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# fit a polynomial of degree to t and y data
def polyfit_D(t, y, degree):
	n = len(y)
	print n
	
	# make a [N x (Degree+1)] matrix
	X = np.zeros((n, degree+1))
	for i in range(n):
		for j in range(degree+1):
			X[i][j] = (t[i])**j
	print X
	
	# using ordinary least squares
	# compute (X^T*X)^(-1) * X^T * y
	XtX = np.dot(np.transpose(X),X)
	inv_XtX = np.linalg.inv(XtX)
	Xty = np.dot(np.transpose(X),y)
	coeff = np.dot(inv_XtX, Xty)
	
	return coeff
	
# plot the polynomial of degree D
def plot_polyn(polyn, D):
	xx = linspace(-1, 1, 200)
	plt.plot(t, y, 'o', xx, polyn(xx),'-r')
	plt.legend(['Data points', str(D)+'th Degree Polynomial'])
	plt.show()
	
# plot MSE vs degree
def plot_MSE(D_vals, MSE_vals):
	# visualize degree vs. MSE 
	plt.plot(D_vals, MSE_vals, 'o')
	plt.title('Plot of the the mean-squared error vs. degree of polynomial')
	plt.xlabel('D values')
	plt.ylabel('Mean-squared error: R(D)')
	plt.show()

# compute mean squared error for estimated polynomial of degree D
def MSE(t, y, polyn, D):
	n = len(y)
	sum = 0.0
	for i in range(0,D):
		sum += (y[i] - polyn(t[i])) ** 2
		
	return sum/n

if __name__ == "__main__":
	t = np.loadtxt('data_problem2.1\\t.dat')
	y_orig = np.loadtxt('data_problem2.1\\y.dat')
	y_fresh = np.loadtxt('data_problem2.1\\yfresh.dat')

	# choose a y data source
	y = y_fresh
	
	n = len(y)
	i = 0
	
	MSE_vals = np.zeros(n-1)
	D_vals = np.zeros(n-1)
	# fit the data with a D degree polynomial
	for D in range(1, n):
		print D
		# compute coefficients for polynomial of degree D
		coeffs = polyfit_D(t,y,D)
		# reverse order of coefficients for poly1d function
		rev_coeffs = np.fliplr([coeffs])[0]
		
		# construct the polynomial for graphing
		polyn = np.poly1d(rev_coeffs) 
		# compute mean squared error
		MSE_vals[i] = MSE(t,y,polyn,D)
		D_vals[i] = D
		i += 1
	
	plot_MSE(D_vals, MSE_vals);