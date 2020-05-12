# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:50:41 2020

@author: nikhil
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from keras import optimizers, losses, metrics

import matplotlib.pyplot as plt

import pandas as pd

def data():
    data = pd.read_csv("train.csv")
    dataX = data.to_numpy()[:,0:-1]
    dataY = data.to_numpy()[:,-1]
    
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=42)
    
    def normalizeData(data, test):
        mean = data.mean(axis=0)
        std = np.std(data, dtype=np.float64)
        data -= mean
        data /= std
        test -= mean
        test /= std
        return data, test
    
    trainX, testX = normalizeData(trainX, testX)  
    
    return trainX, trainY, testX, testY   

def model(trainX, trainY, testX, testY):
    model = Sequential()
    model.add(Dense({{choice([16, 32, 64])}}, input_shape=trainX[0].shape, activation='sigmoid'))
    model.add(Dense({{choice([16, 32, 64])}}, activation='sigmoid'))
    #model.add(BatchNormalization())
    
    model.add(Dense(1))
    
    model.compile(loss='logcosh',
              optimizer=optimizers.Adam(lr={{choice([0.01, 0.001])}}),
              metrics=['mae'])
    
    result = model.fit(trainX, trainY, epochs=80, batch_size=64, validation_split=0.2)
    validation_acc = np.amin(result.history['val_mean_absolute_error'])
    print("Best Validation Accuracy of epoch: ", validation_acc)
    
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())

training = best_model.history.history

val_loss = training['val_loss']
val_accuracy = training['val_mean_absolute_error']
loss = training['loss']
accuracy = training['mean_absolute_error']
epochs = range(1, 80+1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Ensemble model - Creating two data functions and obtaining the ensemble prediction on validation set
# Please uncomment the code below to run ensemble method.

#def data1():
#    data = pd.read_csv("train.csv", header=-1)
#    dataX = data.to_numpy()[0:10000,1:]
#    dataY = data.to_numpy()[0:10000,0]
#    
#    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=42)
#    
#    def normalizeData(data, test):
#        mean = data.mean(axis=0)
#        std = np.std(data, dtype=np.float64)
#        data -= mean
#        data /= std
#        test -= mean
#        test /= std
#        return data, test
#    
#    trainX, testX = normalizeData(trainX, testX)  
#    
#    return trainX, trainY, testX, testY 
#
#def data2():
#    data = pd.read_csv("train.csv", header=-1)
#    dataX = data.to_numpy()[10000:,1:]
#    dataY = data.to_numpy()[10000:,0]
#    
#    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=42)
#    
#    def normalizeData(data, test):
#        mean = data.mean(axis=0)
#        std = np.std(data, dtype=np.float64)
#        data -= mean
#        data /= std
#        test -= mean
#        test /= std
#        return data, test
#    
#    trainX, testX = normalizeData(trainX, testX)  
#    
#    return trainX, trainY, testX, testY  
#
#def fixedmodel(trainX, trainY, testX, testY):
#    model = Sequential()
#    model.add(Dense(16, input_shape=trainX[0].shape, activation='sigmoid'))
#    
#    model.add(Dense(32, activation='sigmoid'))
#    
#    model.add(Dense(1))
#    
#    model.compile(loss='logcosh',
#              optimizer=optimizers.Adagrad(lr={{choice([0.01, 0.001])}}),
#              metrics=['mae'])
#    
#    result = model.fit(trainX, trainY, epochs=80, batch_size=64, validation_split=0.2)
#    validation_acc = np.amin(result.history['val_mean_absolute_error'])
#    print("Best Validation Accuracy of epoch: ", validation_acc)
#    
#    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
#
#
#
#best_run1, best_model1 = optim.minimize(model=fixedmodel, data=data1, algo=tpe.suggest, max_evals=10, trials=Trials())
#best_run2, best_model2 = optim.minimize(model=fixedmodel, data=data2, algo=tpe.suggest, max_evals=10, trials=Trials())
#
#train1, train1Y, _, _ = data1()
#train2, train2Y, _, _ = data2()
#val = np.concatenate(train1[0:2000,:], train2[0:2000,:])
#valY = np.concatenate(train1Y[0:2000,:], train2Y[0:2000,:])
#
#acc=0
#for i in range(0, len(val)):
#    res1 = best_model1.predict(val[i])
#    res2 = best_model2.predict(val[i])
#    res = (res1+res2)/2
#    acc=abs(val[i]-res)
#    
#acc/=2000