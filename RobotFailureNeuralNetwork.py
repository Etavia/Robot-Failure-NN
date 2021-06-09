
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:03:20 2020

@author: Tuomas Ikonen
Based on article "How to Create a Simple Neural Network in Python" by Dr. Michael J. Garbade
"""

import pandas as pd
import numpy as np

#Read data file and change column titles
data=pd.read_csv("lp1.data", sep='\t', skiprows=1, header=None) 

#Divide data in to input data and output data and drop extra columns
inputdata = data
inputdata = data.drop(data.columns[6], axis=1)

outputdata = data
outputdata = data.drop(data.columns[0], axis=1)
outputdata = outputdata.drop(data.columns[1], axis=1)
outputdata = outputdata.drop(data.columns[2], axis=1)
outputdata = outputdata.drop(data.columns[3], axis=1)
outputdata = outputdata.drop(data.columns[4], axis=1)
outputdata = outputdata.drop(data.columns[5], axis=1)

#Read test data file and change column titles
testdata=pd.read_csv("Testdata.csv", sep=';', header=None)

#Divide data in to input data and output data and drop extra columns
testinputdata = testdata
testinputdata = testdata.drop(data.columns[6], axis=1)

testoutputdata = testdata
testoutputdata = testdata.drop(data.columns[0], axis=1)
testoutputdata = testoutputdata.drop(data.columns[1], axis=1)
testoutputdata = testoutputdata.drop(data.columns[2], axis=1)
testoutputdata = testoutputdata.drop(data.columns[3], axis=1)
testoutputdata = testoutputdata.drop(data.columns[4], axis=1)
testoutputdata = testoutputdata.drop(data.columns[5], axis=1)

#Get number of rows and columns
inputdatarows, inputdatacols = inputdata.shape

#Get number of rows and columns
outputdatarows, outputdatacols = outputdata.shape

#Get number of rows and columns
testinputdatarows, testinputdatacols = testinputdata.shape

#Get number of rows and columns
testoutputdatarows, testoutputdatacols = testoutputdata.shape

#Export data in to array
inputarr = np.zeros((inputdatarows,inputdatacols))

#Fill array with data from test data
for x in range(0, inputdatarows):
    for y in range(0, inputdatacols):
        inputarr[x][y] = inputdata[y][x]

#Export data in to array
outputarr = np.zeros((outputdatarows,outputdatacols))
outputarr = outputarr.astype('int32')

#Fill array with data from test data
for x in range(0, outputdatarows):
    outputarr[x][0] = outputdata[6][x]

#Export data in to array
testinputarr = np.zeros((testinputdatarows,testinputdatacols))

#Fill array with data from test data
for x in range(0, testinputdatarows):
    for y in range(0, testinputdatacols):
        testinputarr[x][y] = testinputdata[y][x]

#Export data in to array
testoutputarr = np.zeros((testoutputdatarows,testoutputdatacols))

#Fill array with data from test data
for x in range(0, testoutputdatarows):
    testoutputarr[x][0] = testoutputdata[6][x]

testoutputarr = testoutputarr.astype('int32')

#Scale inputs so that machine learning works better in algorithm   
training_inputs = inputarr * 0.01    
testtraining_inputs = testinputarr * 0.01

training_outputs = outputarr   

class NeuralNetwork():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((inputdatacols, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights, "\n")
     
#training taking place
neural_network.train(training_inputs, training_outputs, 15000)

print("Ending Weights After Training: ")
print(neural_network.synaptic_weights, "\n")

print("Output with test set: ")
print("NN OUTPUT\t\t\tREAL OUTPUT IN DATA")
for x in range(0, testinputdatarows):
    print(neural_network.think(testtraining_inputs[x][:]), "\t\t", testoutputarr[x][:])