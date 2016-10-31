import numpy as np
from numpy import *
import matplotlib.pyplot as plt

# distribution of X with density f
def f(x,c):
	if x >= 0 and x <= 1:
		return c*x*(1-x)
	else:
		return 0

if __name__ == "__main__":
	N = 10000
	c = 2
	x = [None] * N
	i = 0
	samp_count = 0.0
	while i != N:
		y  = np.random.uniform(0,1) 
		# sample u from Unif(0,1)
		u = np.random.uniform(0,1)*c/4 
		if u < f(y, c) : 
			#accept y as sample drawn from f
			x[i] = y
			i = i + 1
			print "accept: ", y
		# else: reject y and return to sampling
		samp_count += 1
	
	
	print samp_count 
	print "E[T]: ", samp_count/N

	plt.hist(x)
	plt.title("Accept-Reject Sampling Histogram")
	plt.show()
