#!/usr/bin/env python
#@author Sharfaraz Salek
#Dimensionality reduction via PCA

import sys
import csv
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

def graph(projectedData):
	#Graph the data
	color = list()
	for diabetic in projectedData[:,0]:
		if diabetic == -1:
			color.append('blue')
		else:
			color.append('red')
	
	plt.scatter(x = projectedData[:,1],
				y = projectedData[:,2],
				c = color)
	plt.show()

def project(rawData):
	#Calculate covariance matrix, eigenvalues and eigenvectors
	covMatrix = np.cov(rawData[:,1:], rowvar=False)
	eigenvalues, eigenvectors = la.eig(covMatrix)
	
	#Projecting on dimension of highest variability
	maxIdx = eigenvalues.argmax() 
	pca = rawData[:,1:].dot(eigenvectors[maxIdx,:]) 
	
	#Projecting on dimension of second highest variability
	eigenvalues2 = np.delete(eigenvalues, maxIdx)
	maxIdx2 = eigenvalues2.argmax()
	
	if maxIdx <= maxIdx2: #if the index of first maximum is equal or less
		maxIdx2 += 1	  #original position would be +1 of new
	pca2 = rawData[:,1:].dot(eigenvectors[:,maxIdx2	])
	
	#combining both vectors with classification
	combined = np.vstack((rawData[:,0], pca, pca2)).T
	return combined


def normalize(rawData):
	#Standardize the dataset
	r, c = rawData.shape
	for col in range(1,c):
		mean = np.mean(rawData[:,col])
		std = np.std(rawData[:,col], ddof=1)
		rawData[:,col] -= mean
		rawData[:,col] /= std

def main(argv):
	#parse csv file for diabetese.csv
	rawData = np.genfromtxt("diabetes.csv", delimiter=",")
	normalize(rawData)
	projectedData = project(rawData)
	graph(projectedData)

	
if __name__ == "__main__":
	main(sys.argv)