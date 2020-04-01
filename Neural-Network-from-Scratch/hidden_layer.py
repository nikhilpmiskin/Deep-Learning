
import pandas as pd
import numpy as np

""" Class for Hidden Layer uses Sigmoid nodes """

class Sigmoid:
    
    def __init__(self,in_dim, out_dim):
        
        self.x = np.zeros((in_dim, 1))
        self.weights = np.random.randn(in_dim,out_dim)
        self.y = np.zeros((out_dim, 1))
        
    def forwardPass(self, x):
        """ Calculation of forward pass through Sigmoid layer """
        
        self.x = x
        
        wt_x = np.inner(np.transpose(self.weights), x)
        
        self.y = np.exp(wt_x) / (1 + np.exp(wt_x))
        
        return self.y
    
    def backwardPass(self, grad_in, lr):
        """ Calculation of backward pass through Sigmoid layer """
        
        sigmoid_grad = ((1 - self.y) * self.y) 
        
        grad_out =  sigmoid_grad * np.sum(grad_in, axis=0)
        
        grad_out = np.outer(grad_out, self.x)
        
        self.weights -= lr*np.transpose(grad_out)
        
        return grad_out
        
