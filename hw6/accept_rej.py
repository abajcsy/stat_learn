import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def p(x,c):
	if x >= 0 and x <= 1:
		return c*x*(1-x)
	else:
		return 0

def q(x):
	

if __name__ == "__main__":
	N = 1000
	M = 2 # M > 1 and bound on p(x)/q(x)
	c = 10
	x = [None] * N
	i = 0
	while i not N:
		x[i]  = q(x)
		u = np.random.uniform(0,1) 
		if u < (p(x[i], c) / M*q(x[i])) : 
			accept x_i
			i = i + 1
		else:
			reject x_i
	