"""
Implementation of problem 4.2
"""
import numpy as np
from numpy import *

class SepSet():
	nodes = []
	compat = []

	def __init__(self, nodes):
		self.nodes = nodes

	# tells you if a given node s is in the separator set
	def indicator(self, s):
		if s in nodes:
			return 1
		else:
			return 0

# Node class that stores the compatability function at that node, 
# the list of neighbor nodes it sends/receives messages to/from,
# the list of incoming messages and outgoing messages
class Clique():
	# list of nodes in this clique
	nodes = []
	# vector for clique's compatibility function
	compat = []
	# list of neighboring cliques
	neigh = []

	# these are dictionaries that map a clique's neighbors 
	# to the message it's sending/receiving  
	in_msgs = {}
	out_msgs = {}

	def __init__(self, compat):
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

		# create list of cliques in junction tree
		for i in range(self.size):
			self.cliques.append(Clique([]))
 
		self.cliques[0].nodes = [0,1,3]
		self.cliques[0].neigh = [1] # stores indices of neighboring cliques in self.cliques

		self.cliques[1].nodes = [1,3,4,7]
		self.cliques[1].neigh = [0,2,4]
		
		self.cliques[2].nodes = [3,6,7]
		self.cliques[2].neigh = [1]

		self.cliques[3].nodes = [1,2,5]
		self.cliques[3].neigh = [4]

		self.cliques[4].nodes = [1,4,5,7]
		self.cliques[4].neigh = [1,3,5]

		self.cliques[5].nodes = [5,7,8]
		self.cliques[5].neigh = [4]

		# initialize messages for all edges uniformly at first
		# for incoming and outgoing edges
		for i in range(self.size):
			self.cliques[i].out_msgs = {}
			self.cliques[i].in_msgs = {}
			for j in self.cliques[i].neigh:
				self.cliques[i].out_msgs[j] = np.array([1,1]) 
				self.cliques[i].in_msgs[j] = np.array([1,1])
	
	# defines the compatibility function for a given edge (or pair of nodes)
	def edge_compat(self, s, t):
		if s == t:
			return 1.0
		else:
			if s == 0 and t == 1:
				return 0.3
			else:
				return 0.5	

	# runs sum-product algorithm on tree
	def sum_prod_factor(self):
		numIter = 20
		for k in range(numIter):
			# for each clique t
			for s in range(self.size):
				# for each clique s that has an edge with t
				for t in self.cliques[s].neigh:	
					self.cliques[t].out_msgs[s] = np.array([0,0])
					# compute for x_s = 0 and x_s = 1
					for idx in range(2):
						# get edge compatibility function 
						edge_compat0 = self.edge_compat(idx,0)
						edge_compat1 = self.edge_compat(idx,1)
						edge_compat = np.array([edge_compat0, edge_compat1])

						# get t's singleton compatibility function 
						compat_t = self.cliques[t].compat
					
						# compute final product of edge and singleton compatibility function
						final_compat = compat_t[idx]*edge_compat

						# compute product of all the received messages from neighbors 
						# that are NOT the one you are sending a message to
						vec_prod = np.array([1,1])
						for u in self.cliques[t].neigh:
							if u != s:
								vec_prod = vec_prod*self.cliques[u].out_msgs[t][idx]
										
						result = final_compat * vec_prod					
						self.cliques[t].out_msgs[s] =  self.cliques[t].out_msgs[s] + result		
					self.cliques[s].in_msgs[t] = self.cliques[t].out_msgs[s]
					
		# compute marginals from formula:
		# p(x_s) = psi(x_s)* prod_over_neighbors(M*_(t->s)(x_s))
		for i in range(self.size):
			marg = np.prod(self.cliques[i].in_msgs.values(),0)
			marg = self.cliques[i].compat*marg
			marg_norm = marg/np.sum(marg)
			print "p(",i, ") = ", marg_norm
			
if __name__ == '__main__':
	t = JunctionTree()	
	t.sum_prod_factor()

