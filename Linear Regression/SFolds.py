#!/usr/bin/env python
#Sharfaraz Salek

from __future__ import division
import sys
import csv
import copy
import random
import argparse
import numpy as np
import pandas as pd
from numpy import linalg as la 

def predict(test, norm, theta, obs):
	#Standardize test set
	r, c = test.shape
	for col in range(1,c-1):
		mean = norm[col-1, 0]
		std = norm[col-1, 1]
		test[:,col] -= mean
		test[:,col] /= std

	#Compute predicted Y values
	mse = 0
	for row in test:
		y = theta[0]
		for col in range(1, c-1):
			y+=row[col]*theta[col]
		mse += pow((row[c-1] - y),2)
		
	#Calcuate RMSE
	mse = mse/r
	rmse = (r/obs)*np.sqrt(mse)
	return rmse

def regression(train):
	#Dimensions of training data
	r, c = train.shape

	#Bias 
	bias = np.ones(r)
	theta = np.zeros(c-1)

	#Creating [1 x] vector
	x = np.ndarray(shape=(r,c-1))
	idxOut = [0, c-1]
	idxIn = [i for i in xrange(c) if i not in idxOut]
	x[:,0] = bias.T
	x[:,1:] = train[:,idxIn]

	#Calculate theta
	temp_theta = (x.T).dot(x)
	temp_theta = la.inv(temp_theta)
	theta = temp_theta.dot(x.T.dot(train[:,-1]))
	
	return theta

def standardize(data):
	#Standardize
	norm = list()
	c = data.shape[1]
	for col in range(1,c-1):
		mean = np.mean(data[:,col])
		std = np.std(data[:,col], ddof=1)
		data[:,col] -= mean
		data[:,col] /= std
		temp = [mean, std]
		norm.append(temp)

	return np.asarray(norm) 

def dataSets(folds, i):
	#Building testing set
	test = pd.DataFrame(folds[i])

	#Build training set
	train = pd.DataFrame()
	for x in range(0, len(folds)):
		if x != i:
			train = train.append(folds[x])
	
	return np.asarray(train), np.asarray(test)

def sfolds(s, rawData):
	#Number of observations
	r, c = rawData.shape

	#Shuffle the data
	np.random.seed(0)
	np.random.shuffle(rawData)
	
	#Create S-Folds
	folds = [[] for i in range(s)]
	df = pd.DataFrame(rawData)
	itemCount = 0
	for i in range(s):
		folds[i] = df.loc[itemCount:itemCount+np.floor(r/s+0.5)-1]
		if s%2 != 0: 
			itemCount += r//s + 1
		else:
			itemCount += r//s
	
	return folds

def main(argv):
	#Parse csv file
	rawData = np.genfromtxt("x06Simple.csv", delimiter=",", skip_header=1)

	if sys.argv[1]>1 and sys.argv[1]<=rawData.shape[0]: 
		#S-Folds cross validation
		folds = sfolds(int(sys.argv[1]), rawData)

		#Calculate rmse leaving one out
		rmse = list()
		for i in range(0, int(sys.argv[1])):

			#Building training set and testing set
			tempFolds = copy.deepcopy(folds)
			train, test = dataSets(tempFolds, i)
			
			#Standardize data set
			normalize = standardize(train)
		
			#Build regression model
			theta = regression(train)

			#Predict
			rmse.append(predict(test, normalize, theta, rawData.shape[0]))

		print np.sum(rmse)

	else:
		print "Not a legal value for S"

if __name__ == "__main__":
	main(sys.argv) 