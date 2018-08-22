import torch
import numpy as np

n1 = 24 #Number of nodes in 1st layer
p = np.arange(16)
p = p/15

d = 1

t=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
   [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
   [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
   [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

print (t.shape)

def net2(A, model_num, lr, niter, p, t, n1):
    #A - Weights and Bises Cell
    #model_num - The nth model in ensemble
    #lr - Learning Rate
    #niter - Number of iterations
    #p - Input LIst
    #t - target outputs
    #-----------------
    #A - Output Weights and Biases 
    #error - Final training error output

    for k in range(1, niter):
        error = 0
	
        for j in range(1, size(p,2)):
            ip = p(j)	    
