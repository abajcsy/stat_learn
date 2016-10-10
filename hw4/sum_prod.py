"""
Implementation of problem 4.2
"""
import numpy as np
from numpy import *
	
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
		
# Tree class represents a tree as a set of nodes. Each node
# has a list of neighbors which represents the possible edges 
# that exist in the tree.
class Tree():
	nodes = []
	size = 0
	
	def __init__(self, max_node):
		self.size = max_node

		for i in range(max_node):
			self.nodes.append(Node(self.single_compat(i)))

		self.nodes[0].neigh = [1,2]
		self.nodes[1].neigh = [0,3,4]
		self.nodes[2].neigh = [0,5]
		self.nodes[3].neigh = [1]
		self.nodes[4].neigh = [1]
		self.nodes[5].neigh = [2]

		# initialize messages for all edges uniformly at first
		# for incoming and outgoing edges
		for i in range(self.size):
			for j in self.nodes[i].neigh:
				self.nodes[i].out_msgs[j] = np.array([1,1]) 
				self.nodes[i].in_msgs[j] = np.array([1,1])
	
	# defines the compatibility function for a given node
	def single_compat(self, s):
		if s%2 == 0:
			return np.array([0.7, 0.3])
		else:
			return np.array([0.1, 0.9])
	
	# defines the compatibility function for a given edge (or pair of nodes)
	def edge_compat(self, s, t):
		if s == t:
			return 1.0
		else:
			return 0.45	
			
	# utility function 	
	def print_tree(self):
		for i in range(self.size):
			compat_f = self.nodes[i].compat
			print "node" + str(i) + ", compat: " + str(compat_f)

	
	def get_msgs(self):
		for i in range(self.size):
			for j in self.nodes[i].neigh:
				self.nodes[i].in_msgs[j] = self.nodes[j].out_msgs[i]
			
	def send_msgs(self):
		# for each node in the tree
		for i in range(self.size):
			# for each neighbor of a node
			for j in self.nodes[i].neigh:
				# get compatibility function of current node at given index 
				compat_f = self.nodes[i].compat
				
				# get edge compatibility function 
				edge_compat = self.edge_compat(i,j)
				
				# compute product of all the received messages from neighbors 
				vec_prod = np.array([1,1])
				for x in self.nodes[i].neigh:
					if x != j:
						vec_prod = vec_prod*self.nodes[i].in_msgs[x]
				
				# update outgoing message for each neighbor node
				self.nodes[i].out_msgs[j] = self.nodes[i].out_msgs[j] + vec_prod * compat_f * edge_compat 

	def sum_prod(self):
		self.send_msgs()
		self.get_msgs()
		
		for i in range(self.size):
			marg = np.prod(self.nodes[i].in_msgs.values(),0)
			marg = self.nodes[i].compat*marg
			marg_norm = marg/np.sum(marg)
			print "p(",i, ") = ", marg_norm
			
if __name__ == '__main__':
	t = Tree(6)	
	t.sum_prod()
