#!/usr/bin/env python
#Sharfaraz Salek

import sys
import csv
import random
import numpy as np
import pandas as pd
from numpy import linalg as la 

def predict(test, norm, theta):
	#standardize test set
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
	rmse = np.sqrt(mse)
	print rmse

def regression(train):
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
	
	print theta
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

def dataSets(rawData):
	#Number of observations
	rows = rawData.shape[0]

	#Shuffle the data
	np.random.seed(0)
	np.random.shuffle(rawData)
	idx = np.random.rand(rows) < 0.667
	
	#Sampling Data
	df = pd.DataFrame(rawData)
	train = df[idx]
	test =  df[~idx]

	return np.asarray(train), np.asarray(test)


def main(argv):
	#Parse csv file
	rawData = np.genfromtxt("x06Simple.csv", delimiter=",", skip_header=1)

	#Building training set and testing set
	train, test = dataSets(rawData)

	#Standardize data set
	norm = standardize(train)
	
	#Build regression model
	theta = regression(train)

	#Predict
	predict(test, norm, theta)


if __name__ == "__main__":
	main(sys.argv) 