"""
Implementation of problem 4.2
"""
import numpy as np
from numpy import *
import sys
import matplotlib.pyplot as plt
import random
	
# Node class that stores the compatability function at that node, 
# the list of neighbor nodes it sends/receives messages to/from,
# the list of incoming messages and outgoing messages
class Node():
	# stores vector for node's singleton compatibility function
	compat = []
	# stores list of neighbors 
	neigh = []

	# these are dictionaries that map a node's neighbors 
	# to the message it's sending/receiving  
	in_msgs = {}
	out_msgs = {}

	def __init__(self, compat):
		self.compat = compat

	# debugging functions
	def __repr__(self):
		string = "{neigh: " + str(self.neigh) + ", in_msgs: " + str(self.in_msgs) + ", out_msgs: " + str(self.out_msgs) + "}\n"
		return string

	def __str__(self):
		string = "{neigh: " + str(self.neigh) + ", in_msgs: " + str(self.in_msgs) + ", out_msgs: " + str(self.out_msgs) + "}\n"
		return string
		
# Tree class represents a tree as a set of nodes. Each node
# has a list of neighbors which represents the possible edges 
# that exist in the tree.
class Tree():
	nodes = []
	size = 0
	
	def __init__(self, max_node):
		self.size = max_node

		for i in range(self.size):
			self.nodes.append(Node(self.single_compat(i)))

		# make toroidal graph structure 
		for i in range(self.size):
			# look at neighbors up, down, left, right
			self.nodes[i].neigh = [None]*4
				
		# index row and column into 2D grid
		for i in range(7):
			for j in range(7):
				# get number of node from location in 2D grid
				node_idx = i*7+j
				neigh = self.get_neigh(i,j)
				# for each node in graph, set it's neighbor's values to 1 
				for n in range(len(neigh)):
					ni = neigh[n][0]
					nj = neigh[n][1]
					self.nodes[node_idx].neigh[n] = ni*7+nj

		for node in range(self.size):
			self.nodes[node].neigh = [7*(node//7) + (node+1)%7, 
													7*(node//7) + (node-1)%7,
													(node+7)%49,
													(node-7)%49]
					
		#for i in range(self.size):
		#	print "node (",i,") neigh:",self.nodes[i].neigh
			
		# initialize messages for all edges uniformly at first
		# for incoming and outgoing edges
		for i in range(self.size):
			self.nodes[i].out_msgs = {}
			self.nodes[i].in_msgs = {}
			for j in self.nodes[i].neigh:
				self.nodes[i].out_msgs[j] = np.array([np.random.rand(),np.random.rand()]) 
				self.nodes[i].in_msgs[j] = np.array([np.random.rand(),np.random.rand()])
	
	# get indices of neighbors of node at index i,j
	def get_neigh(self, i, j):
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
	
	# defines the compatibility function for a given node
	def single_compat(self, s):
		return np.array([1, 1])
	
	# defines the compatibility function for a given edge (or pair of nodes)
	def edge_compat(self, s, t, gamma):
		if s == t:
			return np.exp(gamma)
		else:
			return np.exp(-gamma)
			
	# runs sum-product algorithm on tree
	def sum_prod(self, gamma, numIter):
		for k in range(numIter):
			# for each node t
			for s in range(self.size):
				# for each node s that has an edge with t
				for t in self.nodes[s].neigh:	
					self.nodes[t].out_msgs[s] = np.array([0,0])
					# compute for x_s = 0 and x_s = 1
					for idx in range(2):
						# get edge compatibility function 
						edge_compat0 = self.edge_compat(idx,0, gamma)
						edge_compat1 = self.edge_compat(idx,1, gamma)
						edge_compat = np.array([edge_compat0, edge_compat1])

						# get t's singleton compatibility function 
						compat_t = self.nodes[t].compat
					
						# compute final product of edge and singleton compatibility function
						final_compat = compat_t[idx]*edge_compat

						# compute product of all the received messages from neighbors 
						# that are NOT the one you are sending a message to
						vec_prod = np.array([1,1])
						for u in self.nodes[t].neigh:
							if u != s:
								vec_prod = vec_prod*self.nodes[u].out_msgs[t][idx]	
								
						result = final_compat*vec_prod	
						self.nodes[t].out_msgs[s] = result/np.sum(result)
					self.nodes[s].in_msgs[t] = self.nodes[t].out_msgs[s]
				
		# compute marginals from formula:
		# p(x_s) = psi(x_s)* prod_over_neighbors(M*_(t->s)(x_s))
		p = np.zeros((49,2))
		for i in range(self.size):
			marg = np.prod(self.nodes[i].in_msgs.values(),0)
			marg = self.nodes[i].compat*marg
			marg_norm = marg/np.sum(marg)
			p[i][0] = marg_norm[0]
			p[i][1] = marg_norm[1]
			print "p(",i, ") = ", marg_norm
		
		return p
			
if __name__ == '__main__':
	
	gamma = float(sys.argv[1])
	numIter = 50
	
	p = np.zeros(100)

	for i in range(10):
		t = Tree(49)	
		marg_norm = t.sum_prod(i/100, numIter)
		p[i] = marg_norm[0][0]-0.5

	print p
	x = np.arange(100)/100
	plt.plot(x,np.abs(p))
	plt.plot((.3465, .3465), (-0.6, 0.6), 'red')
	plt.show()
	
	print 'gamma: ', gamma

