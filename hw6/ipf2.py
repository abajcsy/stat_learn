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
	
	for n in range(20):
		for i in range(num_c):
			(s,t) = cliques[i]
			if(s == 0):
				if(t==1):
					# compute new psi_01[0][0]
					psi_01_00_new = psi[0][0][0][0][0] * mu_hat[0][0][0]/mu_old[0][0][0]
					# deal with psi_01(x_0 = 0)(x_1 = 0)
					psi[0][0][0][0][0] = psi_01_00_new
					psi[0][0][1][0][0] = psi_01_00_new
					psi[0][0][0][1][0] = psi_01_00_new
					psi[0][0][1][1][0] = psi_01_00_new
					
					# compute new psi_01[0][1]
					psi_01_01_new = psi[0][1][0][0][0] * mu_hat[0][0][1]/mu_old[0][0][1]
					# deal with psi_01(x_0 = 0)(x_1 = 1)
					psi[0][1][0][0][0] = psi_01_01_new 
					psi[0][1][1][0][0] = psi_01_01_new
					psi[0][1][0][1][0] = psi_01_01_new
					psi[0][1][1][1][0] = psi_01_01_new
					
					# compute new psi_01[1][0]
					psi_01_10_new = psi[1][0][0][0][0] * mu_hat[0][1][0]/mu_old[0][1][0]
					# deal with psi_01(x_0 = 1)(x_1 = 0)
					psi[1][0][0][0][0] = psi_01_10_new
					psi[1][0][1][0][0] = psi_01_10_new
					psi[1][0][0][1][0] = psi_01_10_new
					psi[1][0][1][1][0] = psi_01_10_new
					
					# compute new psi_01[1][1]
					psi_01_11_new = psi[1][1][0][0][0] * mu_hat[0][1][1]/mu_old[0][1][1]
					# deal with psi_01(x_0 = 1)(x_1 = 1)
					psi[1][1][0][0][0] = psi_01_11_new
					psi[1][1][1][0][0] = psi_01_11_new
					psi[1][1][0][1][0] = psi_01_11_new
					psi[1][1][1][1][0] = psi_01_11_new
					
					# update mu_old
					mu_old_01 = np.array((2,2))
					# take product along all psi's
					mu_old_01[0][0] = np.prod(psi[0][0][0][0], 0)  + np.prod(psi[0][0][1][0], 0) + np.prod(psi[0][0][0][1], 0) + np.prod(psi[0][0][1][1], 0)   
					mu_old_01[0][1] = np.prod(psi[0][1][0][0], 0)  + np.prod(psi[0][1][1][0], 0) + np.prod(psi[0][1][0][1], 0) + np.prod(psi[0][1][1][1], 0)  
					mu_old_01[1][0] = np.prod(psi[1][0][0][0], 0)  + np.prod(psi[1][0][1][0], 0) + np.prod(psi[1][0][0][1], 0) + np.prod(psi[1][0][1][1], 0)   
					mu_old_01[1][1] = np.prod(psi[1][1][0][0], 0)  + np.prod(psi[1][1][1][0], 0) + np.prod(psi[1][1][0][1], 0) + np.prod(psi[1][1][1][1], 0)   
					
					# normalize 
					norm_const = mu_old_01[0][0] + mu_old_01[0][1] + mu_old_01[1][0] + mu_old_01[1][1]
					mu_old_01 = mu_old_01/norm_const
					
					mu_old[0] = mu_old_01
					
				else if(t == 3):
					# compute new psi_03[0][0]
					psi_03_00_new = psi[0][0][0][0][3] * mu_hat[3][0][0]/mu_old[3][0][0]
					# deal with psi_03(x_0 = 0)(x_3 = 0)
					psi[0][0][0][0][3] = psi_03_00_new
					psi[0][0][1][0][3] = psi_03_00_new
					psi[0][1][0][0][3] = psi_03_00_new
					psi[0][1][1][0][3] = psi_03_00_new
					
					# compute new psi_03[0][1]
					psi_03_01_new = psi[0][0][0][1][3] * mu_hat[3][0][1]/mu_old[3][0][1]
					# deal with psi_01(x_0 = 0)(x_3 = 1)
					psi[0][0][0][1][3] = psi_03_01_new
					psi[0][0][1][1][3] = psi_03_01_new
					psi[0][1][0][1][3] = psi_03_01_new
					psi[0][1][1][1][3] = psi_03_01_new
					
					# compute new psi_03[1][0]
					psi_03_10_new = psi[1][0][0][0][3] * mu_hat[3][1][0]/mu_old[3][1][0]
					# deal with psi_01(x_0 = 1)(x_3 = 0)
					psi[1][0][0][0][3] = psi_03_10_new
					psi[1][0][1][0][3] = psi_03_10_new
					psi[1][1][0][0][3] = psi_03_10_new
					psi[1][1][1][0][3] = psi_03_10_new
					
					# compute new psi_03[1][1]
					psi_03_11_new = psi[1][0][0][1][3] * mu_hat[3][1][1]/mu_old[3][1][1]
					# deal with psi_01(x_0 = 1)(x_1 = 1)
					psi[1][0][0][1][3] = psi_03_11_new
					psi[1][0][1][1][3] = psi_03_11_new
					psi[1][1][0][1][3] = psi_03_11_new
					psi[1][1][1][1][3] = psi_03_11_new
					
					# update mu_old
					mu_old_03 = np.array((2,2))
					# take product along all psi's
					mu_old_03[0][0] = np.prod(psi[0][0][0][0], 0)  + np.prod(psi[0][0][1][0], 0) + np.prod(psi[0][1][0][0], 0) + np.prod(psi[0][1][1][0], 0)   
					mu_old_03[0][1] = np.prod(psi[0][0][0][1], 0)  + np.prod(psi[0][0][1][1], 0) + np.prod(psi[0][1][0][1], 0) + np.prod(psi[0][1][1][1], 0)  
					mu_old_03[1][0] = np.prod(psi[1][0][0][0], 0)  + np.prod(psi[1][0][1][0], 0) + np.prod(psi[1][1][0][0], 0) + np.prod(psi[1][1][1][0], 0)   
					mu_old_03[1][1] = np.prod(psi[1][0][0][1], 0)  + np.prod(psi[1][1][0][1], 0) + np.prod(psi[1][0][1][1], 0) + np.prod(psi[1][1][1][1], 0)   
					
					# normalize 
					norm_const = mu_old_03[0][0] + mu_old_03[0][1] + mu_old_03[1][0] + mu_old_03[1][1]
					mu_old_03 = mu_old_03/norm_const
					
					mu_old[3] = mu_old_03					
			else if(s == 1 and t == 2):
					# compute new psi_12[0][0]
					psi_12_00_new = psi[0][0][0][0][1] * mu_hat[1][0][0]/mu_old[1][0][0]
					# deal with psi_12(x_1 = 0)(x_2 = 0)
					psi[0][0][0][0][1] = psi_12_00_new
					psi[1][0][0][0][1] = psi_12_00_new
					psi[0][0][0][1][1] = psi_12_00_new
					psi[1][0][0][1][1] = psi_12_00_new
					
					# compute new psi_12[0][1]
					psi_12_01_new = psi[0][0][1][0][1] * mu_hat[1][0][1]/mu_old[1][0][1]
					# deal with psi_12(x_1 = 0)(x_2 = 1)
					psi[0][0][1][0][1] = psi_12_01_new 
					psi[1][0][1][0][1] = psi_12_01_new
					psi[0][0][1][1][1] = psi_12_01_new
					psi[1][0][1][1][1] = psi_12_01_new
					
					#-------------------------TODO FROM HERE------------------------------
					
					# compute new psi_01[1][0]
					psi_01_10_new = psi[1][0][0][0][1] * mu_hat[1][1][0]/mu_old[1][1][0]
					# deal with psi_01(x_0 = 1)(x_1 = 0)
					psi[1][0][0][0][0] = psi_01_10_new
					psi[1][0][1][0][0] = psi_01_10_new
					psi[1][0][0][1][0] = psi_01_10_new
					psi[1][0][1][1][0] = psi_01_10_new
					
					# compute new psi_01[1][1]
					psi_01_11_new = psi[1][1][0][0][0] * mu_hat[1][1][1]/mu_old[1][1][1]
					# deal with psi_01(x_0 = 1)(x_1 = 1)
					psi[1][1][0][0][0] = psi_01_11_new
					psi[1][1][1][0][0] = psi_01_11_new
					psi[1][1][0][1][0] = psi_01_11_new
					psi[1][1][1][1][0] = psi_01_11_new
					
					# update mu_old
					mu_old_01 = np.array((2,2))
					# take product along all psi's
					mu_old_01[0][0] = np.prod(psi[0][0][0][0], 0)  + np.prod(psi[0][0][1][0], 0) + np.prod(psi[0][0][0][1], 0) + np.prod(psi[0][0][1][1], 0)   
					mu_old_01[0][1] = np.prod(psi[0][1][0][0], 0)  + np.prod(psi[0][1][1][0], 0) + np.prod(psi[0][1][0][1], 0) + np.prod(psi[0][1][1][1], 0)  
					mu_old_01[1][0] = np.prod(psi[1][0][0][0], 0)  + np.prod(psi[1][0][1][0], 0) + np.prod(psi[1][0][0][1], 0) + np.prod(psi[1][0][1][1], 0)   
					mu_old_01[1][1] = np.prod(psi[1][1][0][0], 0)  + np.prod(psi[1][1][1][0], 0) + np.prod(psi[1][1][0][1], 0) + np.prod(psi[1][1][1][1], 0)   
					
					# normalize 
					norm_const = mu_old_01[0][0] + mu_old_01[0][1] + mu_old_01[1][0] + mu_old_01[1][1]
					mu_old_01 = mu_old_01/norm_const
					
					mu_old[1] = mu_old_01
			else if(s == 2 and t == 3):
				
				
			psi[i] = psi[i]*mu_hat[i]/mu_old[i]
	
	
	print psi

if __name__ == "__main__":
	data = np.loadtxt('Pairwise.dat')
	problem2i()