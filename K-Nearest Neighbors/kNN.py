#!/usr/bin/env python
#Author: Sharfaraz Salek

from __future__ import division
import sys
import csv
import copy
import random
import numpy as np
import pandas as pd
from numpy import linalg as la 
from scipy import stats

def errorMeasure(labels, test):
	#Initialize variables
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	row, col = test.shape
	
	#Calculate hits
	col -= 1
	for obs in range(0, labels.shape[0]):
		if labels[obs] == 1 and test[obs, col] == 1:
			TP += 1
		elif labels[obs] == 1 and test[obs, col] == 0:
			FP += 1
		elif labels[obs] == 0 and test[obs, col] == 0:
			TN += 1
		elif labels[obs] == 0 and test[obs, col] == 1:
			FN += 1

	#Classifier evaluation
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	fmeasure = (2*precision*recall)/(precision+recall)
	accuracy = (TP+TN)/(TP+TN+FP+FN)

	#Print results
	print precision
	print recall
	print fmeasure
	print accuracy

def kNN(norm, train, test, k):
	#Standardize Test Set
	r, c = test.shape
	for col in range(0,c-1):
		mean = norm[col, 0]
		std = norm[col, 1]
		test[:,col] -= mean
		test[:,col] /= std

	#Initialize label vector
	row, col = test.shape
	label = np.zeros(test.shape[0])
	
	for obs in range(0, test.shape[0]):
		#Calculate distance between all training observations
		temp = copy.deepcopy(test[obs,:col-1])
		temp = np.tile(temp,(train.shape[0],1))
		distances = np.sum(np.absolute(np.subtract(train[:,:col-1], temp)), axis=1)

		#Sort k-elements by distances and return index
		rowIdx = np.argpartition(distances, k)
		neighbors = [train[i,col-1] for i in rowIdx[:k]]

		#Create array with predicted labels with mode
		label[obs] = stats.mode(neighbors)[0]

	return label

def standardize(data):
	#Standardize
	norm = list()
	c = data.shape[1]
	for col in range(0,c-1):
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
	rawData = np.genfromtxt("spambase.data", delimiter=",")
	k = int(argv[1])

	#Building training set and testing set
	train, test = dataSets(rawData)

	#Standardize data set
	norm = standardize(train)

	#K Nearest Neighbors
	labels = kNN(norm, train, test, k)

	#Prediction Accuracy
	errorMeasure(labels, test)

if __name__ == "__main__":
	main(sys.argv) 