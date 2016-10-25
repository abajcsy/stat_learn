import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def init_matrices(num_c):
	list = [None] * num_c
	length = len(list)
	for i in range(length):
		list[i] = np.ones((2,2))
	return list

def compute_mu_hat(psi, mu, data, clique_nodes):
	mu_hat = mu
	(s,t) = clique_nodes
	for i in range(2):
		for j in range(2):
			# compute sum
			sum = 0
			for k in range(30):
				sum += psi[i][j]*(data[s][k] == j)*(data[t][k] == i)
			mu_hat[i][j] = 1/30.0 * sum
	return mu_hat
	
if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')
	print data

	cliques = [(0,1), (1,2), (2,3), (0,3)]
	num_c = len(cliques)
	print num_c
	
	psi = init_matrices(num_c)
	
	mu_hat =  init_matrices(num_c)
	mu_old = []
	
	for i in range(num_c):
		mu_old = mu_hat
		mu_hat = compute_mu_hat(psi[i], mu_old[i], data, cliques[i])
		psi[i] = psi[i]*mu_hat[i]/mu_old[i]
	
	print psi
	print mu_hat