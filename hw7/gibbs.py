import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random

# get indices of neighbors of node at index i,j
def get_neigh(i, j):
	# look at neighbors up, down, left, right
	neigh = [[i-1,j], [i+1,j], [i,j-1], [i,j+1]]
	# deal with edge cases
	if i == 0: 			# upper left of graph
		neigh[0] = [6,j]
	elif i == 6: 		# upper right of graph
		neigh[1] = [0,j]
	if j == 0: 			# lower left of graph
		neigh[2] = [i,6]
	elif j == 6: 		# lower right of graph
		neigh[3] = [i,0]
	return neigh

def make_donut_graph(num_nodes):
	# make a 49x49 sized matrix
	# for each node n in {1,...49}, store 1 for each corresponding neighbor
	graph = np.zeros((num_nodes, num_nodes))

	# index row and column into 2D grid
	for i in range(7):
		for j in range(7):
			# get number of node from location in 3D grid
			node_idx = i*7+j
			neigh = get_neigh(i,j)
			# for each node in graph, set it's neighbor's values to 1 
			for n in range(len(neigh)):
				ni = neigh[n][0]
				nj = neigh[n][1]
				neigh_idx = ni*7+nj
				graph[node_idx, neigh_idx] = 1
	return graph

def gibbs(graph, num_nodes, theta_st, theta_s, burn_in, num_samps):
	# store for each of the {1,...49} nodes, it's corresponding samples
	samples = np.zeros((num_nodes, num_samps))
	X_est = np.ones((num_nodes,1))	
	# make X_est values somewhat random 1 or -1 (?)

	for i in range(burn_in + num_samps):
		# permute ordering of indices (?)
		for n in range(num_nodes):
			# get neighbors of node
			n_neigh = X_est[graph[n] != 0] 
			p = theta_s[n] + theta_st * np.sum(n_neigh)
			rand_num = np.random.randn()
			if rand_num < np.exp(p)/(np.exp(p)+np.exp(-p)):
				X_est[n] = 1
			else:
				X_est[n] = -1
		# record samples after burn-in period
		if(i > burn_in):
			print samples[:, i-burn_in]
			print X_est
			samples[:, i-burn_in] = X_est
		
	return samples
	
if __name__ == "__main__":
	num_nodes = 49

	# edge and node compatibility functions
	theta_st = 0.25
	theta_s = [(-1.0)**s for s in range(1,num_nodes+1)]
	print len(theta_s)

	# make donut graph
	graph = make_donut_graph(num_nodes)
	print graph

	burn_in = 1000
	num_samps = 5000

	# run gibbs sampler 
	gibbs_samps = gibbs(graph, num_nodes, theta_st, theta_s, burn_in, num_samps)
	print gibbs_samps
