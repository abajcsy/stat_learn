import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def init_matrices(clique_nodes):
	list = [None] * len(clique_nodes)
	i = 0
	for n in clique_nodes:
		c_size = len(n)
		list[i] = np.ones((c_size,c_size))
		i = i+1
	return list

def compute_mu_hat(mu, data, clique_nodes):
	mu_hat = mu
	for j in range(2):
		for k in range(2):
			# compute sum
			sum = 0
			for i in range(30):
				(s,t) = clique_nodes
				sum += (data[s][i] == j)*(data[t][i] == k)
			mu_hat[j][k] = 1/30.0 * sum
	return mu_hat
	
def compute_mu_old(psi, cliques, curr_clique):
	
	
if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')

	cliques = [(0,1), (1,2), (2,3), (0,3)]
	num_c = len(cliques)
	
	psi = init_matrices(cliques)
	mu_hat = init_matrices(cliques)
	mu_old = init_matrices(cliques)
	
	# compute mu_hat for each clique
	for i in range(num_c):
		mu_hat[i] = compute_mu_hat(mu_hat[i], data, cliques[i])
	
	for n in range(10):
		for i in range(num_c):
			psi[i] = psi[i]*mu_hat[i]/mu_old[i]
	