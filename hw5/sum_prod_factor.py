"""
Implementation of problem 4.2
"""
import numpy as np
from numpy import *

class Compat():
	nodes = []
	table = []

	def __init__(self, nodes, table):
		self.nodes = nodes
		self.table = table

	def get_indices_from_nodes(node_list):
		# if Compat has nodes [1,3], then it has a 2x2 table of compat function
		# given node_list [1], it returns a list [0] for the index of node [1] values

		# if Compat has nodes [1,4,7], then it has a 2x2x2 table of compat function
		# given node_list [1,4], it returns a list [0,1] for the index of node [1,4] values		
		indices = []
		for i in node_list:
			

	def multiply(compat):
		union_nodes = self.nodes
		# find all common nodes from two compat functions
		for m in compat.nodes:
			if m in union_nodes:
				continue
			else:
				union_nodes.append(m)

		dim_tuple = ()
		for i in len(union_nodes):
			dim_tuple = dim_tuple + (2,)

		new_table = np.array(dim_tuple)
		
		#for each node_s in self.nodes
			# index into the self.table to get (0,0), (0,1), (1,0), etc. element
		#for each node_t in compat.nodes
			# index into compat.table to get ()
			new_table[]
				
	def divide(compat):
		

class SepSet():
	nodes = []
	compat = []

	def __init__(self, nodes):
		self.nodes = nodes

		# make separator set compatibility function initialized to unity
		if len(self.nodes) == 2:
			compat = np.array((2,2))
				for i in range(2):
					for j in range(2):
						compat[i][j] = 1
		else:
			compat = np.array((2,2,2))
			for i in range(2):
				for j in range(2):
					for j in range(2):
						compat[i][j][k] = 1
					
# Node class that stores the compatability function at that node, 
# the list of neighbor nodes it sends/receives messages to/from,
# the list of incoming messages and outgoing messages
class Clique():
	# list of nodes in this clique
	nodes = []
	# vector for clique's compatibility function
	compat = None
	# list of neighboring cliques
	neigh = []

	def __init__(self, nodes, neigh, compat):
		self.nodes = nodes
		self.neigh = neigh
		# make clique compatibility function
		self.compat = compat
		
# Tree class represents a tree as a set of nodes. Each node
# has a list of neighbors which represents the possible edges 
# that exist in the tree.
class JunctionTree():
	cliques = []
	sep_sets = []
	size = 0
	sep_size = 0
	
	def __init__(self):
		self.size = 6
		self.sep_size = 4

		# create list of separator sets
		sep_sets = [SepSet([1,3]), SepSet([3,7]), SepSet([1,4,7]), SepSet([1,5]), SepSet([5,7])]

		# create list of cliques in junction tree, each with standard compatibility function
		c01 = Compat([0,1], np.array([1.0, 0.3][0.5, 1.0]))
		c03 = Compat([0,3], np.array([1.0, 0.3][0.5, 1.0]))
		c14 = Compat([1,4], np.array([1.0, 0.3][0.5, 1.0]))
		c34 = Compat([3,4], np.array([1.0, 0.3][0.5, 1.0]))
		c47 = Compat([4,7], np.array([1.0, 0.3][0.5, 1.0]))
		c36 = Compat([3,6], np.array([1.0, 0.3][0.5, 1.0]))
		c67 = Compat([6,7], np.array([1.0, 0.3][0.5, 1.0]))
		c12 = Compat([1,2], np.array([1.0, 0.3][0.5, 1.0]))
		c25 = Compat([2,5], np.array([1.0, 0.3][0.5, 1.0]))
		c45 = Compat([4,5], np.array([1.0, 0.3][0.5, 1.0]))
		c58 = Compat([5,8], np.array([1.0, 0.3][0.5, 1.0]))
		c78 = Compat([7,8], np.array([1.0, 0.3][0.5, 1.0]))

		self.cliques.append(Clique([0,1,3], [1], c01.multiply(c03)))
		self.cliques.append(Clique([1,3,4,7], [0,2,4], c14.multiply(c34).multiply(c47)))
		self.cliques.append(Clique([3,6,7], [1], c36.multiply(c67)))
		self.cliques.append(Clique([1,2,5], [4], c12.multiply(c25)))
		self.cliques.append(Clique([1,4,5,7], [1,3,5], c45))
		self.cliques.append(Clique([5,7,8], [4], c58.multiply(c78)))

	def init_sep_set():
		
	
if __name__ == '__main__':
	t = JunctionTree()	

