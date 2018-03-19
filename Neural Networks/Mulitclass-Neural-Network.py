#!/usr/bin/env python

from __future__ import division
import sys
import csv
import copy
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def predict(test, beta, theta, classes):
	#Create output layer
	testRows, testCols = test.shape
	outTest = np.zeros(testRows)
	Y = np.zeros((testRows, classes.shape[0]))
	for z in range(testRows):
		Y[z,int(test[z,-1]-1)] = 1

	#Add bias to input data
	inputLayer = np.ones((testRows, testCols))
	inputLayer[:,1:] = test[:,:-1]

	#Run test data through network
	H = inputLayer.dot(beta)
	H = 1/(1+np.exp(-H))
	O = H.dot(theta)
	O = 1/(1+np.exp(-O))

	#Testing Accuracy
	gotRight = 0
	outTest = np.zeros(testRows)

	#Classify test data on probability
	for x in range(testRows):
		maxIdx = np.argmax(O[x]) #Index of label with highest probability
		outTest[x] = maxIdx+1 #Labels are 1 indexed

		#Calculate accuracy
		if outTest[x] == test[x,-1]:
			gotRight += 1

	testAccuracy = gotRight/testRows
	print 'Testing Accuracy:', testAccuracy*100

def ann(train, classes):
	#Initialize variables
	iterations = 1000
	hiddenLayers = 20
	accuracy = np.zeros((iterations,2))
	accuracy[:,0] = np.arange(iterations)

	#Create output layer
	trainRows, trainCols = train.shape 
	outTrain = np.zeros(trainRows)
	Y = np.zeros((trainRows, classes.shape[0]))
	for z in range(trainRows):
		Y[z,int(train[z,-1]-1)] = 1
	
	#Bias input to hidden layer
	inputLayer = np.ones((trainRows, trainCols))
	inputLayer[:,1:] = train[:,:-1]

	#Initialize beta and theta
	beta = np.random.uniform(-1,1,size=(trainCols, hiddenLayers))
	theta = np.random.uniform(-1,1,size=(hiddenLayers, classes.shape[0]))
	learnRate = 0.5/trainRows

	#Train network to learn beta and theta
	for i in range(iterations):
		#Propogate Forward
		gotRight = 0
		H = inputLayer.dot(beta)
		H = 1/(1+np.exp(-H))
		O = H.dot(theta)
		O = 1/(1+np.exp(-O))
		
		#Error at output layer
		deltaOut = np.subtract(Y,O)
		
		#Update weights from hidden layer to output
		theta = theta + learnRate*(H.T.dot(deltaOut))

		#Compute error at the hidden layer
		deltaHidden = np.multiply(deltaOut.dot(theta.T), H)
		deltaHidden = np.multiply(deltaHidden, np.subtract(1,H))

		#Update weights from input layer to hidden
		beta = beta + learnRate*(inputLayer.T.dot(deltaHidden))

		#Classify data on probability
		for x in range(trainRows):
			maxIdx = np.argmax(O[x])
			outTrain[x] = maxIdx+1 #Labels are 1 indexed

			#Calculate accuracy
			if outTrain[x] == train[x,-1]:
				gotRight += 1

		#Update accurancy matrix
		accuracy[i,1] = gotRight/trainRows

	#Tesing accuracy graphs
	plt.plot(accuracy[:,0], accuracy[:,1])
	plt.show()

	#Return weights
	return beta, theta

def standardize(train, test):
	#Standardize
	c = train.shape[1]
	for col in range(c-1):
		mean = np.mean(train[:,col])
		std = np.std(train[:,col], ddof=1)
		
		#Training data
		train[:,col] -= mean
		train[:,col] /= std
		
		#Testing data
		test[:,col] -= mean
		test[:,col] /= std

def dataSets(rawData):
	#Number of observations
	rows = rawData.shape[0]

	#Shuffle the data
	np.random.seed(0)
	np.random.shuffle(rawData)
	idx = np.random.rand(rows) < 0.67
	
	#Sampling Data
	df = pd.DataFrame(rawData)
	train = df[idx]
	test =  df[~idx]

	return np.asarray(train), np.asarray(test)

def cleanData(rawData):
	#Delete first 2 rows and second last column
	labels = pd.DataFrame(rawData[2:,-1])
	rawData = pd.DataFrame(rawData[2:,:-2])
	rawData = pd.concat([rawData, labels], axis=1)

	return np.asarray(rawData)

def main(argv):
	#Parse csv file
	rawData = np.genfromtxt("CTG.csv", delimiter=",")
	rawData = cleanData(rawData)

	#Building training set and testing set
	train, test = dataSets(rawData)

	#Standardize data set
	standardize(train, test)

	#Classes in dataset
	classes = np.unique(rawData[:,-1])

	#Train Neural Network
	beta, theta = ann(train, classes)

	#Predict labels for testing set
	predict(test, beta, theta, classes)

if __name__ == "__main__":
	main(sys.argv) 