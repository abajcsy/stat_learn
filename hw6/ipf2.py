import numpy as np
from numpy import *
import matplotlib.pyplot as plt

#----------------------------------------------------------#
#----------------- PROBLEM 2(i)  ------------------------#
#----------------------------------------------------------#
def compute_mu_hat_2i(mu, data, clique_nodes):
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

def init_mu_2i(clique_nodes):
	list = [None] * len(clique_nodes)
	i = 0
	for n in clique_nodes:
		c_size = len(n)
		list[i] = np.ones((c_size,c_size))
		i = i+1
	return list

def init_S_2i():
	S = np.ones((2,2,2,2,4))
	return S
	
def problem2i():
	cliques = [(0,1), (1,2), (2,3), (0,3)]
	num_c = len(cliques)

	S = init_S_2i()
	mu_hat = init_mu_2i(cliques)
	
	# compute mu_hat for each clique
	for i in range(num_c):
		mu_hat[i] = compute_mu_hat_2i(mu_hat[i], data, cliques[i])
	
	for n in range(10):
		for c in range(num_c):
			if(c == 0): # clique (0,1)
				sum00 = np.sum(np.prod(S[0,0,:,:],2))
				sum01 = np.sum(np.prod(S[0,1,:,:],2))
				sum10 = np.sum(np.prod(S[1,0,:,:],2))
				sum11 = np.sum(np.prod(S[1,1,:,:],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[0,0,:,:,c] = S[0,0,:,:,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[0,1,:,:,c] = S[0,1,:,:,c]*mu_hat[c][0][1]/(sum01/norm)
				S[1,0,:,:,c] = S[1,0,:,:,c]*mu_hat[c][1][0]/(sum10/norm) 
				S[1,1,:,:,c] = S[1,1,:,:,c] *mu_hat[c][1][1]/(sum11/norm) 
				
			elif(c == 3): # clique (0,3)
				sum00 = np.sum(np.prod(S[0,:,:,0],2))
				sum01 = np.sum(np.prod(S[0,:,:,1],2))
				sum10 = np.sum(np.prod(S[1,:,:,0],2))
				sum11 = np.sum(np.prod(S[1,:,:,1],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[0,:,:,0,c] = S[0,:,:,0,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[0,:,:,1,c] = S[0,:,:,1,c] *mu_hat[c][0][1]/(sum01/norm) 
				S[1,:,:,0,c] = S[1,:,:,0,c]*mu_hat[c][1][0]/(sum10/norm)  
				S[1,:,:,1,c] = S[1,:,:,1,c]*mu_hat[c][1][1]/(sum11/norm)  
	
			elif(c == 1): # clique (1,2)
				sum00 = np.sum(np.prod(S[:,0,0,:],2))
				sum01 = np.sum(np.prod(S[:,0,1,:],2))
				sum10 = np.sum(np.prod(S[:,1,0,:],2))
				sum11 = np.sum(np.prod(S[:,1,1,:],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[:,0,0,:,c] = S[:,0,0,:,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[:,0,1,:,c] = S[:,0,1,:,c]*mu_hat[c][0][1]/(sum01/norm)  
				S[:,1,0,:,c] = S[:,1,0,:,c]*mu_hat[c][1][0]/(sum10/norm)   
				S[:,1,1,:,c] = S[:,1,1,:,c]*mu_hat[c][1][1]/(sum11/norm) 
				
			elif(c == 2): # clique (2,3)
				sum00 = np.sum(np.prod(S[:,:,0,0],2))
				sum01 = np.sum(np.prod(S[:,:,0,1],2))
				sum10 = np.sum(np.prod(S[:,:,1,0],2))
				sum11 = np.sum(np.prod(S[:,:,1,1],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[:,:,0,0,c] = S[:,:,0,0,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[:,:,0,1,c] = S[:,:,0,1,c]*mu_hat[c][0][1]/(sum01/norm)  
				S[:,:,1,0,c] = S[:,:,1,0,c]*mu_hat[c][1][0]/(sum10/norm) 
				S[:,:,1,1,c] = S[:,:,1,1,c]*mu_hat[c][1][1]/(sum11/norm)  
					
	print "psi_01 = \n", S[:,:,0,0,0], "\n"
	print "psi_12 = \n", S[0,:,:,0,1], "\n"
	print "psi_23 = \n", S[0,0,:,:,2], "\n"
	print "psi_03 = \n", S[:,0,0,:,3], "\n"

	
#----------------------------------------------------------#
#----------------- PROBLEM 2(ii)  ------------------------#
#----------------------------------------------------------#

def compute_mu_hat_2ii(mu, data, clique_nodes):
	mu_hat = mu
	for j in range(2):
		for k in range(2):
				for m in range(2):
					# compute sum
					sum = 0
					for i in range(30):
						(s,t,u) = clique_nodes
						sum += (data[s][i] == j)*(data[t][i] == k)*(data[u][i] == m)
					mu_hat[j][k][m] = 1/30.0 * sum
	return mu_hat

def init_mu_2ii(clique_nodes):
	list = [None] * len(clique_nodes)
	list[0] = np.ones((2,2,2))
	return list

def init_S_2ii():
	S = np.ones((2,2,2,1))
	return S
	
def problem2ii():
	cliques = [(0,1,2)]
	num_c = len(cliques)

	S = init_S_2ii()
	mu_hat = init_mu_2ii(cliques)
	
	# compute mu_hat for each clique
	for i in range(num_c):
		mu_hat[i] = compute_mu_hat_2ii(mu_hat[i], data, cliques[i])
	print mu_hat[0]
	
	for n in range(10):
		for c in range(num_c):
			if(c == 0): # clique (0,1,2)
				sum000 = np.sum(np.prod(S[0,0,0],0))
				sum001 = np.sum(np.prod(S[0,0,1],0))
				sum010 = np.sum(np.prod(S[0,1,0],0))
				sum100 = np.sum(np.prod(S[1,0,0],0))
				sum101 = np.sum(np.prod(S[1,0,1],0))
				sum110 = np.sum(np.prod(S[1,1,0],0))
				sum011 = np.sum(np.prod(S[0,1,1],0))
				sum111 = np.sum(np.prod(S[1,1,1],0))
				
				norm = sum000 + sum001 + sum010 + sum100 + sum101 + sum110 + sum011 + sum111
 			
				S[0,0,0,c] = S[0,0,0,c]*mu_hat[c][0][0][0]/(sum000/norm) 
				S[0,0,1,c] = S[0,0,1,c]*mu_hat[c][0][0][1]/(sum001/norm) 
				S[0,1,0,c] = S[0,1,0,c]*mu_hat[c][0][1][0]/(sum010/norm)
				S[1,0,0,c] = S[1,0,0,c]*mu_hat[c][1][0][0]/(sum100/norm) 
				S[1,0,1,c] = S[1,0,1,c]*mu_hat[c][1][0][1]/(sum101/norm) 
				S[1,1,0,c] = S[1,1,0,c] *mu_hat[c][1][1][0]/(sum110/norm) 
				S[0,1,1,c] = S[0,1,1,c] *mu_hat[c][0][1][1]/(sum011/norm) 
				S[1,1,1,c] = S[1,1,1,c] *mu_hat[c][1][1][1]/(sum111/norm) 
					
	print "psi_012 = \n", S[:,:,:,0], "\n"
if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')
	problem2i()
	problem2ii()
	#problem2iii()