import numpy as np
from numpy import *
import matplotlib.pyplot as plt


#----------------------------------------------------------#
#----------------- UTILITY FUNC  ------------------------#
#----------------------------------------------------------#	

def init_S(num_cliques):
	S = np.ones((2,2,2,2,num_cliques))
	return S
	
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
	
#----------------------------------------------------------#
#----------------- PROBLEM 2(i)  ------------------------#
#----------------------------------------------------------#	

# compute P(x1...xd)  = (1/Z) * prod(psi_st(x_s, x_t))	
def get_likelihood_2i(S, cliques):
	prod = np.array((2,2))
	num_cliques = len(cliques)
	
	# take product over all psi(s,t)

	# multiply clique (0,1)*(1,2)
	a = S[:,:,:,0,0]
	b = S[:,:,:,0,1]
	c = a*b
	# clique (2,3)
	d = S[0,:,:,:,2]
	print "a"
	print a
	print "mult a*b"
	print a*b
	print "b"
	print b
	print "adghfadf"
	print np.prod(a,b,2)
	#prod0 = S[:,:,:,0,0]*S[:,:,:,0,1]
	
def problem2i():
	cliques = [(0,1), (1,2), (2,3), (0,3)]
	num_c = len(cliques)

	S = init_S(num_c)
	mu_hat = init_mu(cliques)
	
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
					
	print "PROBLEM 2(i) RESULTS:"
	print "psi_01 = \n", S[:,:,0,0,0], "\n"
	print "psi_12 = \n", S[0,:,:,0,1], "\n"
	print "psi_23 = \n", S[0,0,:,:,2], "\n"
	print "psi_03 = \n", S[:,0,0,:,3], "\n"

	get_likelihood_2i(S, cliques)
	
#----------------------------------------------------------#
#----------------- PROBLEM 2(ii)  ------------------------#
#----------------------------------------------------------#

def problem2ii():
	cliques = [(0,1), (0,2), (0,3), (1,2)]
	num_c = len(cliques)

	S = init_S(num_c)
	mu_hat = init_mu(cliques)
	
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
				
			elif(c == 2): # clique (0,3)
				sum00 = np.sum(np.prod(S[0,:,:,0],2))
				sum01 = np.sum(np.prod(S[0,:,:,1],2))
				sum10 = np.sum(np.prod(S[1,:,:,0],2))
				sum11 = np.sum(np.prod(S[1,:,:,1],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[0,:,:,0,c] = S[0,:,:,0,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[0,:,:,1,c] = S[0,:,:,1,c] *mu_hat[c][0][1]/(sum01/norm) 
				S[1,:,:,0,c] = S[1,:,:,0,c]*mu_hat[c][1][0]/(sum10/norm)  
				S[1,:,:,1,c] = S[1,:,:,1,c]*mu_hat[c][1][1]/(sum11/norm)  
	
			elif(c == 1): # clique (0,2)
				sum00 = np.sum(np.prod(S[0,:,0,:],2))
				sum01 = np.sum(np.prod(S[0,:,1,:],2))
				sum10 = np.sum(np.prod(S[1,:,0,:],2))
				sum11 = np.sum(np.prod(S[1,:,1,:],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[0,:,0,:,c] = S[0,:,0,:,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[0,:,1,:,c] = S[0,:,1,:,c]*mu_hat[c][0][1]/(sum01/norm)  
				S[1,:,0,:,c] = S[1,:,0,:,c]*mu_hat[c][1][0]/(sum10/norm)   
				S[1,:,1,:,c] = S[1,:,1,:,c]*mu_hat[c][1][1]/(sum11/norm) 
				
			elif(c == 3): # clique (1,2)
				sum00 = np.sum(np.prod(S[:,0,0,:],2))
				sum01 = np.sum(np.prod(S[:,0,1,:],2))
				sum10 = np.sum(np.prod(S[:,1,0,:],2))
				sum11 = np.sum(np.prod(S[:,1,1,:],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[:,0,0,:,c] = S[:,0,0,:,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[:,0,1,:,c] = S[:,0,1,:,c] *mu_hat[c][0][1]/(sum01/norm)  
				S[:,1,0,:,c] = S[:,1,0,:,c] *mu_hat[c][1][0]/(sum10/norm) 
				S[:,1,1,:,c] = S[:,1,1,:,c] *mu_hat[c][1][1]/(sum11/norm)  
					
	print "\nPROBLEM 2(ii) RESULTS:"
	print "psi_01 = \n", S[:,:,0,0,0], "\n"
	print "psi_02 = \n", S[:,0,:,0,1], "\n"
	print "psi_03 = \n", S[:,0,0,:,2], "\n"
	print "psi_12 = \n", S[0,:,:,0,3], "\n"
	
#----------------------------------------------------------#
#----------------- PROBLEM 2(iii)  ------------------------#
#----------------------------------------------------------#
	
def problem2iii():
	cliques = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
	num_c = len(cliques)

	S = init_S(num_c)
	mu_hat = init_mu(cliques)
	
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
				
			elif(c == 1): # clique (0,2)
				sum00 = np.sum(np.prod(S[0,:,0,:],2))
				sum01 = np.sum(np.prod(S[0,:,1,:],2))
				sum10 = np.sum(np.prod(S[1,:,0,:],2))
				sum11 = np.sum(np.prod(S[1,:,1,:],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[0,:,0,:,c] = S[0,:,0,:,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[0,:,1,:,c] = S[0,:,1,:,c]*mu_hat[c][0][1]/(sum01/norm)  
				S[1,:,0,:,c] = S[1,:,0,:,c]*mu_hat[c][1][0]/(sum10/norm)   
				S[1,:,1,:,c] = S[1,:,1,:,c]*mu_hat[c][1][1]/(sum11/norm) 
				
			elif(c == 2): # clique (0,3)
				sum00 = np.sum(np.prod(S[0,:,:,0],2))
				sum01 = np.sum(np.prod(S[0,:,:,1],2))
				sum10 = np.sum(np.prod(S[1,:,:,0],2))
				sum11 = np.sum(np.prod(S[1,:,:,1],2))
				norm = sum00 + sum01 + sum10 + sum11
				
				S[0,:,:,0,c] = S[0,:,:,0,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[0,:,:,1,c] = S[0,:,:,1,c] *mu_hat[c][0][1]/(sum01/norm) 
				S[1,:,:,0,c] = S[1,:,:,0,c]*mu_hat[c][1][0]/(sum10/norm)  
				S[1,:,:,1,c] = S[1,:,:,1,c]*mu_hat[c][1][1]/(sum11/norm)  
	
			elif(c == 3): # clique (1,2)
				sum00 = np.sum(np.prod(S[:,0,0,:],2))
				sum01 = np.sum(np.prod(S[:,0,1,:],2))
				sum10 = np.sum(np.prod(S[:,1,0,:],2))
				sum11 = np.sum(np.prod(S[:,1,1,:],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[:,0,0,:,c] = S[:,0,0,:,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[:,0,1,:,c] = S[:,0,1,:,c] *mu_hat[c][0][1]/(sum01/norm)  
				S[:,1,0,:,c] = S[:,1,0,:,c] *mu_hat[c][1][0]/(sum10/norm) 
				S[:,1,1,:,c] = S[:,1,1,:,c] *mu_hat[c][1][1]/(sum11/norm)  
			
			elif(c == 4): # clique (1,3)
				sum00 = np.sum(np.prod(S[:,0,:,0],2))
				sum01 = np.sum(np.prod(S[:,0,:,1],2))
				sum10 = np.sum(np.prod(S[:,1,:,0],2))
				sum11 = np.sum(np.prod(S[:,1,:,1],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[:,0,:,0,c] = S[:,0,:,0,c] *mu_hat[c][0][0]/(sum00/norm) 
				S[:,0,:,1,c] = S[:,0,:,1,c] *mu_hat[c][0][1]/(sum01/norm)  
				S[:,1,:,0,c]= S[:,1,:,0,c] *mu_hat[c][1][0]/(sum10/norm) 
				S[:,1,:,1,c] = S[:,1,:,1,c] *mu_hat[c][1][1]/(sum11/norm)  
			
			elif(c == 5): # clique (2,3)
				sum00 = np.sum(np.prod(S[:,:,0,0],2))
				sum01 = np.sum(np.prod(S[:,:,0,1],2))
				sum10 = np.sum(np.prod(S[:,:,1,0],2))
				sum11 = np.sum(np.prod(S[:,:,1,1],2))
				norm = sum00 + sum01 + sum10 + sum11
			
				S[:,:,0,0,c] = S[:,:,0,0,c]*mu_hat[c][0][0]/(sum00/norm) 
				S[:,:,0,1,c] = S[:,:,0,1,c]*mu_hat[c][0][1]/(sum01/norm)  
				S[:,:,1,0,c] = S[:,:,1,0,c]*mu_hat[c][1][0]/(sum10/norm) 
				S[:,:,1,1,c] = S[:,:,1,1,c]*mu_hat[c][1][1]/(sum11/norm)  
					
	print "\nPROBLEM 2(iii) RESULTS:"
	print "psi_01 = \n", S[:,:,0,0,0], "\n"
	print "psi_02 = \n", S[:,0,:,0,1], "\n"
	print "psi_03 = \n", S[:,0,0,:,2], "\n"
	print "psi_12 = \n", S[0,:,:,0,3], "\n"
	print "psi_13 = \n", S[0,:,0,:,4], "\n"
	print "psi_23 = \n", S[0,0,:,:,5], "\n"

if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')
	problem2i()
	#problem2ii()
	#problem2iii()