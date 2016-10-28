import numpy as np
from numpy import *
import matplotlib.pyplot as plt

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

def init_mu(clique_nodes):
	list = [None] * len(clique_nodes)
	i = 0
	for n in clique_nodes:
		c_size = len(n)
		list[i] = np.ones((c_size,c_size))
		i = i+1
	return list

def init_S():
	S = np.ones((2,2,2,2,4))
	return S
	
def problem2i():
	cliques = [(0,1), (1,2), (2,3), (0,3)]
	num_c = len(cliques)

	S = init_S()
	mu_hat = init_mu(cliques)
	mu_old = init_mu(cliques)
	
	# compute mu_hat for each clique
	for i in range(num_c):
		mu_hat[i] = compute_mu_hat(mu_hat[i], data, cliques[i])
	
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
					
		print 'bla'
		print S[0,0,0,0,0]
		print S[0,0,0,0,1]
		print S[0,0,0,0,2]
		print S[0,0,0,0,3]

if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')
	problem2i()