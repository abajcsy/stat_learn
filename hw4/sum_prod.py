"""
Implementation of problem 4.2
"""
import numpy as np
from numpy import *

class Node():
	compat = []
	neigh = []

	in_msgs = []
	out_msgs = []

	def __init__(self, compat):
		self.compat = compat

	# computes marginal for this node
	#def marginal(self):
		
	
class Tree():
	nodes = []
	size = 0

	def __init__(self, max_node):
		self.size = max_node

		for i in range(max_node):
			self.nodes.append(Node(single_compat(i)))

		self.nodes[0].neigh = [1,2]
		self.nodes[1].neigh = [0,3,4]
		self.nodes[2].neigh = [0,5]
		self.nodes[3].neigh = [1]
		self.nodes[4].neigh = [1]
		self.nodes[5].neigh = [2]

	def print_tree(self):
		for i in range(self.size):
			compat_f = self.nodes[i].compat
			print "node" + str(i) + ", compat: " + str(compat_f)

	def send_msgs(self):
		for i in range(self.size):
			for j in nodes[i].neigh:
		

def edge_compat(s,t):
	if s == t:
		return 1.0
	else:
		return 0.45	

def single_compat(s):
	if s%2 == 0:
		return np.array([0.7, 0.3])
	else:
		return np.array([0.1, 0.9])


if __name__ == '__main__':
	t = Tree(6)	
	t.print_tree()


