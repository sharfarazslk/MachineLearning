#!/usr/bin/env python

#Sharfaraz Salek
#Kmeans Clustering

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

def purity(rawData, plotcluster, k):
	#Group labeled data with appropriate clusters
	obs_cluster = np.zeros(shape=(2,k))
	for i in range(0, k):
		for obs in range(0, rawData.shape[0]):
			if rawData[obs][0] == -1 and plotcluster[obs] == i:
				obs_cluster[i, 0] += 1
			elif rawData[obs][0] == 1 and plotcluster[obs] == i:
				obs_cluster[i, 1] += 1

	#Purity of Clusters
	cluster_purtiy = np.zeros(k)
	total_purity = 0
	for i in range(0,k):
		cluster_purtiy[i] = np.amax(obs_cluster[i])/np.sum(obs_cluster[i,:])
		total_purity += np.amax(obs_cluster[i])
	
	#Total purity
	total_purity = total_purity/rawData.shape[0]
	print total_purity


def plot(rawData, plotcluster, xcol, ycol, initVectors):

	#Set color matrix for each feature
	color_center = ['red', 'blue', 'cyan', 'yellow', 
					'green', 'magenta', 'black']
	colors = list()
	for diabetic in plotcluster:
			colors.append(color_center[diabetic])
	
	#Graph observations
	plt.scatter(x=rawData[:, xcol],
			 	y=rawData[:, ycol],
			 	c=colors,
			 	marker='x')

	#Graph centers of each cluster
	plt.scatter(x=initVectors[:, xcol],
				y=initVectors[:, ycol],
				s= 150,
				c=color_center[0:rawData.shape[1]])
	plt.show()

def kmeans(rawData, k, xcol, ycol):
	#Rows and columns of matrix
	r, c = rawData.shape

	#Randomly select k-initial reference vectors
	np.random.seed(0)
	idx = np.random.randint(0, r, k)
	initVectors = rawData[idx]
	prevMean = np.ndarray(shape=(initVectors.shape[0], initVectors.shape[1]))
	plotcluster = np.zeros(r)
	Termination = 100

	#Clustering the data
	while Termination > pow(2,-23):
		clusters = []
		for obs in rawData:
			#Calculate Euclidian distance
			euclidian = np.zeros(k)
			counter = 0
			for i in initVectors:
				euclidian[counter] = la.norm(obs - i) 
				counter += 1
			idx = np.argmin(euclidian)
			clusters.append(idx)
		
		#Associate cluster data with rest
		clusters = np.asarray(clusters)
		x = [[] for i in range(k)]
		for i in range(0,r):
			x[int(clusters[i])].append(rawData[i])

		#Find new center
		prevMean = np.copy(initVectors)
		for i in range(0,k):
			temp = np.asarray(x[i])
			temp_rows = temp.shape[0]
			for col in range(0, c):
				col_average = np.sum(temp[:,col])/temp_rows
				initVectors[i][col] = col_average
		
		#Keeps track of latest clustering matrix
		plotcluster = clusters

		#Termination condition
		Termination = 0
		for i in range(0,k):
			Termination += np.sum(np.absolute(np.subtract(prevMean[i],initVectors[i])))
		
	#Plot results of kmeans clustering
	plot(rawData, plotcluster, xcol, ycol, initVectors)
	return plotcluster
	

def normalize(rawData):
	#Standardize the dataset
	r, c = rawData.shape
	for col in range(1,c):
		mean = np.mean(rawData[:,col])
		std = np.std(rawData[:,col], ddof=1)
		rawData[:,col] -= mean
		rawData[:,col] /= std

def main(argv):
	#parse csv file
	rawData = np.genfromtxt("diabetes.csv", delimiter=",")
	normalize(rawData)
	
	#Shuffle the data
	np.random.seed(0)
	np.random.shuffle(rawData)
	
	#Caluculate K-Means and plot graph
	clusterData = kmeans(rawData[:,6:8], 2, 1, 0)

	#Calculate Purity of clusters
	purity(rawData, clusterData, 2)

if __name__ == "__main__":
	main(sys.argv)