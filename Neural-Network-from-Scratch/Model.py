

import pandas as pd
import numpy as np
from hidden_layer import Sigmoid
from output_layer import Softmax

""" Model file for overall forward propagation, back propagation and evaluation """

class Model:
    
    def __init__(self):
        self.layers=[]
        self.in_dim= 0
        self.out_dim=0
        self.results = {'Training_Loss' : [], 'Training_Accuracy' : [],
                        'Validation_Loss' : [], 'Validation_Accuracy' : []}
    
    def build(self, in_dim, out_dim, hiddenLayerNodesList):
        """ Method to build model """
        
        np.random.seed(6)
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        for i in range(0,len(hiddenLayerNodesList)):
            layerInputDim = in_dim if i == 0 else hiddenLayerNodesList[i-1] 
            self.layers.append(Sigmoid(layerInputDim, hiddenLayerNodesList[i]))
        
        self.layers.append(Softmax(hiddenLayerNodesList[i],out_dim))
        
        
    def forwardPropagarion(self, x):
        """ Overall Forward Propagation """
        
        for layer in self.layers:
            x = layer.forwardPass(x)
        
        yhat = x
        
        return yhat
    
    def getLossGrad(self, yhat, y):
        return yhat - y
    
    def evaluateLoss(self, yhat, y):
        
        logp = -1*np.log(yhat[y == 1])
        loss = np.sum(logp) 
        
        return loss/len(yhat)
    
    def evaluateAccuracy(self, yhat, y):
        
        hits = 0
        for j in range(0,len(y)):
            if np.argmax(yhat[j]) == np.argmax(y[j]):
                hits+=1
        accuracy = hits/len(y)
        
        return np.round(accuracy*100, 2)
    
    def backwardPropagation(self, y_pred, y , lr):
        """ Overall BackPropagation """
        
        grad = self.getLossGrad(y_pred, y)
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backwardPass(grad, lr)
            
    def evaluateModel(self, x, y):
        """ Method to evaluate model """
        
        yhat = np.zeros(y.shape)
        for i in range(len(x)):
            yhat[i] = self.forwardPropagarion(x[i])
        
        loss = self.evaluateLoss(yhat, y)
        accuracy = self.evaluateAccuracy(yhat,y)
        
        return loss, accuracy
            
    def fit(self, trainX, trainY, valX, valY, lr, epochs, decay):
        """ Method to train model """
        
        for epoch in range(epochs):
            print("Epoch: " + str(epoch))
            for i in range(len(trainX)):
                yhat= self.forwardPropagarion(trainX[i])
                self.backwardPropagation(yhat, trainY[i], lr)
                lr = lr*(1. / (1. + decay))
                
            training_loss , training_accuracy = self.evaluateModel(trainX, trainY)
            validation_loss , validation_accuracy = self.evaluateModel(valX, valY)
            print({'Training_Loss': training_loss , 
                                     'Training_Accuracy': training_accuracy , 
                                     'Validation_Loss': validation_loss, 
                                     'Validation_Accuracy': validation_accuracy})
            self.results['Training_Loss'].append(training_loss)
            self.results['Training_Accuracy'].append(training_accuracy)
            self.results['Validation_Loss'].append(validation_loss)
            self.results['Validation_Accuracy'].append(validation_accuracy)
            
        return self.results
        
        