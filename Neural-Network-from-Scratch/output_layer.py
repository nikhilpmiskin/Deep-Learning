
import numpy as np

""" Class for Output Layer uses Softmax nodes """

class Softmax:
    
    def __init__(self,in_dim, out_dim):
        
        self.x = np.zeros((in_dim, 1))
        self.weights = np.random.randn(in_dim,out_dim)
        self.y = np.zeros((out_dim, 1))
        
    def forwardPass(self, x):
        """ Calculation of forward pass through Softmax layer """
        
        self.x = x
        
        wtx = np.inner(np.transpose(self.weights), x)
        e = np.exp(wtx) 
        
        self.y = e / np.sum(e, axis=-1, keepdims=True)
       
        return self.y
    
    def backwardPass(self, grad_in, lr):
        """ Calculation of backward pass and gradient update through Softmax layer """
        
        y_diagMat = np.diag(self.y)
        y2 = self.y * self.y
        
        softmax_grad= y_diagMat - y2
        grad_out = np.inner(softmax_grad, grad_in)
        
        grad_out = np.outer(grad_out , self.x)
        
        self.weights -= lr*np.transpose(grad_out)
         
        return grad_out
        
    
