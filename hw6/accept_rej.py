import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# distribution of X with density f
def f(x,c):
	if x >= 0 and x <= 1:
		return c*x*(1-x)
	else:
		return 0

# distribution of Y with denisty g
def g(x):
	return np.random.uniform(0,1) 
	
if __name__ == "__main__":
	N = 1000
	M = 2 # M > 1 and bound on f(x)/g(x)
	c = 6
	x = [None] * N
	i = 0
	while i != N:
		y  = g(x)
		# sample u from Unif(0,1)
		u = np.random.uniform(0,1) 
		if u < (f(y, c) / M*g(y)) : 
			#accept y as sample drawn from f
			x[i] = y
			i = i + 1
			print "accept: ", y
		# else: reject y and return to sampling
	
	plt.hist(x,N)
	plt.title("Accept-Reject Sampling Histogram")
	plt.show()