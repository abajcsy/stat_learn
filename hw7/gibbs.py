import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random

# get indices of neighbors of node at index i,j
def get_neigh(i, j):
	# look at neighbors up, down, left, right
	neigh = [None]*4
	neigh[0] = [i-1,j]
	neigh[1] = [i+1,j]
	neigh[2] = [i,j-1]
	neigh[3] = [i,j+1]
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
	# initialize X_est values to random -1, 0 or 1 value
	X_est = np.ones((num_nodes,1))
	for i in range(num_nodes):
		X_est[i] = np.random.randint(-1,1) 

	for i in range(burn_in + num_samps):
		for n in range(num_nodes):
			# get neighbors of node
			n_neigh = X_est[graph[n] != 0] 
			p = theta_s[n] + theta_st * np.sum(n_neigh)
			rand_num = np.random.rand()
			if rand_num < np.exp(p)/(np.exp(p)+np.exp(-p)):
				X_est[n] = 1
			else:
				X_est[n] = -1
		# record samples after burn-in period
		if(i > burn_in):
			for j in range(num_nodes):
				samples[j, i-burn_in] = X_est[j]
		
	return samples
	
def gibbs_mean(gibbs_samps, num_samps, num_nodes):
	mean = np.zeros((num_nodes, 1))
	for i in range(num_nodes):
		mean[i] = sum(gibbs_samps[i,:])/num_samps
	mean_formatted = np.zeros((7,7))
	
	n = 0
	for i in range(7):
		for j in range(7):
				mean_formatted[i][j] = mean[n]
				n += 1
	return mean_formatted
	
def naive_mean(graph, num_nodes, theta_st, theta_s):
	X_est = np.random.rand(num_nodes,1)	
	thresh = 0.0000000001
	curr_diff = 10
	
	# keep iterating until converged
	while(curr_diff > thresh):
		for n in range(num_nodes):
			prev_X_est = X_est
			n_neigh = X_est[graph[n] != 0] 
			p = theta_s[n] + theta_st * np.sum(n_neigh)
			X_est[n] = (np.exp(2*p) - 1)/(np.exp(2*p) + 1)
		diff = X_est - prev_X_est 
		curr_diff = np.sum(diff)/num_nodes 
		
	# reshape to be 7x7 matrix
	X_est_formatted = np.zeros((7,7))
	n = 0
	for i in range(7):
		for j in range(7):
				X_est_formatted[i][j] = X_est[n]
				n += 1
	return X_est_formatted
	
def compute_l1_dist(gibbs_mean, naive_samps):
	dist = 0
	for i in range(7):
		for j in range(7):
			dist += abs(naive_samps[i][j] - gibbs_mean[i][j])
	dist /= 49.0
	return dist
	
if __name__ == "__main__":
	num_nodes = 49

	# edge and node compatibility functions
	theta_st = 0.25
	theta_s = [(-1.0)**s for s in range(1,num_nodes+1)]

	# make donut graph
	graph = make_donut_graph(num_nodes)

	burn_in = 1000
	num_samps = 5000

	# run gibbs sampler 
	print "(a) Gibbs Sampler Results"
	gibbs_samps = gibbs(graph, num_nodes, theta_st, theta_s, burn_in, num_samps)
	gibbs_mean = gibbs_mean(gibbs_samps, num_samps, num_nodes)
	print gibbs_mean
	
	# run naive mean field sampler
	print "(b) Naive Mean Field Results"
	naive_samps = naive_mean(graph, num_nodes, theta_st, theta_s)
	print naive_samps
	dist = compute_l1_dist(gibbs_mean, naive_samps)
	print "Average l1 dist btwn mean field and gibbs estimates: ", dist
