#!/usr/bin/env python

from __future__ import division
import sys
import csv
import copy
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy import stats

def confusionMatrix(predictedLabels, classes, test):
	#Initialize confusion matrix
	row, col = test.shape
	classNum = classes.shape[0]
	conMatrix = np.zeros((classNum, classNum))

	#Calculate predivted vs actual count for each class
	for i in range(row):
		conMatrix[int(predictedLabels[i]-1), int(test[i,-1]-1)] += 1

	#Find % hits
	conMatrix = np.divide(conMatrix, row)
	print conMatrix

def errorMeasure(labelList, test):
	#Initialize variables
	rightLabel = 0
	row, col = test.shape
	predictedLabels = np.zeros(row)
	labelList = np.array(labelList).T
	
	#Select most frequent label
	for i in range(row):
		#Find mode of classification
		mode, count = stats.mode(labelList[i])
		tempCount = count
		tempMode = mode
		modeList = list()

		#If there are more than one modes
		while True:
			temp = np.delete(labelList[i], tempMode)
			tempmode, tempcount = stats.mode(labelList[i])
			if tempcount == count:
				modeList.append(modeList)
			else:
				break
		
		if len(modeList) == 0:
			predictedLabels[i] = mode
		else:
			idx = np.random.randint(0,high=len(modeList)-1)
			predictedLabels[i] = modeList[idx]		

	#Calculate accuracy
	for i in range(row):
		if predictedLabels[i] == test[i,-1]:
			rightLabel += 1

	accuracy = rightLabel/row
	print accuracy
	return predictedLabels

def svm(train, test):
	#Support Vector Machine Training
	clf = SVC(kernel='linear', C=1, gamma=1)
	clf.fit(train[:,:-1], train[:,-1])
	
	#Prediction
	label = clf.predict(test[:,:-1])
	return label

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
	idx = np.random.rand(rows) < 0.667
	
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

def ovoDataset(classes, train, i, j):
	#Initiliaze lists
	trainovo = list()
	
	#Filter out necessary classes from training set
	for z in range(train.shape[0]):
		if train[z,-1] == classes[i] or train[z,-1] == classes[j]:
			trainovo.append(train[z])

	return np.asarray(trainovo)

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
	labelList = list()

	#One-vs-one label
	for i in range(0, classes.shape[0]):
		for j in range(i+1, classes.shape[0]):
			
			#Create one-vs-one datasets
			trainovo = ovoDataset(classes, train, i, j)
			
			#Support Vector Machines
			labels = svm(trainovo, test)
			labelList.append(labels)

	#Accuracy of classification
	predictedLabels = errorMeasure(labelList, test)

	#Confusion matrix
	confusionMatrix(predictedLabels, classes, test)
	
if __name__ == "__main__":
	main(sys.argv) 