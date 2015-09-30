from pylab import *
from numpy import *
import math
import numpy
import random

from random import choice


## No. of Gaussians/ components
noOfG = 2


## Reading file o Data Sructues ##
gauss2 = []
file2Gauss = open("2gaussian.txt", 'r')
for line in file2Gauss:
	line = line.strip('\r\n')
	line = line.split(" ")
	tempStr = '\''+str(line[0])+';'+str(line[1])+'\''
	gauss2.append((numpy.matrix(tempStr)))
file2Gauss.close()
GaussDataLen2 = len(gauss2)
print GaussDataLen2


### Purpose : To calculate the probability of the data given mean and conV
### Contract: (number, matrix, matrix) -> number
### Returns : phi  
def calculatePhi(data, mean, conV):
	expTerm = (-0.5)*(numpy.transpose(data - mean))*(numpy.linalg.inv(conV))*(data - mean)
	phi = math.exp(expTerm)/((2*pi)*math.sqrt(math.fabs(numpy.linalg.det(conV))))
	return phi


### Purpose : To calculate the likelihood given weights, mean and conV
### Contract: (matrix, matrix, matrix) -> number
### Returns : likelihood  
def CalLikelihood(weight, mean, conV):
	likelihood = 0.0
	for data in range(GaussDataLen2):
		term1 = 0.0
		for component in range(noOfG):
			term1 += weight[component] * calculatePhi(gauss2[data], mean[component], conV[component])
		likelihood += math.log(term1)
	likelihood /= float(GaussDataLen2)
	return likelihood


### Purpose : To Initialize the data
### Contract: () -> (number, matrix, matrix, matrix)
### Returns : [likelihood, weight,mean,conV]  
def Initialize():
	mean = [numpy.matrix('0.5;0.5'), numpy.matrix('8.0;4.0')]
	conV = [numpy.matrix('1.0,0.0;0.0,1.0'), numpy.matrix('1.5,0.5;0.5,1.0')]
	weight = [0.6, 0.4]
	likelihood = CalLikelihood(weight,mean,conV)
	# print likelihood
	return likelihood, weight, mean, conV


### Purpose : To calculate the gamma and nComponent given weight, mean and conV
### Contract: (number, matrix, matrix) -> [number[noOfG],number[noOfG]]
### Returns : [gamma, nComponent]  
def EStep(weight, mean, conV):	
	gamma = []
	for data in range(GaussDataLen2):
		gamma.append([])
		for component in range(noOfG):
			gamma[data].append(0.0)
	nComponent = []
	for component in range(noOfG):
		nComponent.append(0.0)
	for data in range(GaussDataLen2):
		total = 0.0
		for component in range(noOfG):
			gamma[data][component] = weight[component] * calculatePhi(gauss2[data], mean[component], conV[component])
			total += gamma[data][component]
		for component in range(noOfG):
			gamma[data][component] /= total
	for component in range(noOfG):
		for data in range(GaussDataLen2):
			nComponent[component] += gamma[data][component]
	return gamma,nComponent


### Purpose : To update the data given weight, mean and conV
### Contract: (number, matrix, matrix, number[noOfG],number[noOfG]) -> [matrix,matrix,matrix]
### Returns : [newWeight,newMean,newConV] 
def MStep(weight, mean, conV, gamma, nComponent):
	newWeight = []
	newMean = [numpy.matrix('0.0;0.0'), numpy.matrix('0.0;0.0')]
	newConV = []
	for component in range(noOfG):
		newWeight.append(0.0)
		newConV.append(0.0)
	for component in range(noOfG):
		newWeight[component] = nComponent[component]/GaussDataLen2
	for component in range(noOfG):
		for data in range(GaussDataLen2):
			newMean[component] += gamma[data][component]*gauss2[data]
		newMean[component] /= nComponent[component]
	for component in range(noOfG):
		for data in range(GaussDataLen2):
			newConV[component] += gamma[data][component]*((gauss2[data] - newMean[component])*numpy.transpose(gauss2[data] - newMean[component]))
		newConV[component] /= nComponent[component]
	return newWeight, newMean, newConV


### Purpose :	To run the EM. This is the function that ties everthing together. 
###				It takes threshold, that governs when do we have to stop converging
### Contract: 	(number) -> ()
### Returns : 	()  
def EM(threshold):
	likelihood,weight,mean,conV = Initialize()
	count = 1
	while('true'):
		gammaANDnComponent = EStep(weight,mean,conV)
		newValuesWMConV = MStep(weight,mean,conV,gammaANDnComponent[0], gammaANDnComponent[1])
		newLikelihood = CalLikelihood(newValuesWMConV[0],newValuesWMConV[1],newValuesWMConV[2])
		if(math.fabs(newLikelihood - likelihood) <= threshold):
			print count, newLikelihood
			print "newWeight", newValuesWMConV[0], newValuesWMConV[0][0]+newValuesWMConV[0][1]
			print "newMean", newValuesWMConV[1]
			print "newConV", newValuesWMConV[2]
			break
		print count, likelihood
		print "Weight", newValuesWMConV[0], newValuesWMConV[0][0]+newValuesWMConV[0][1]
		print "Mean", newValuesWMConV[1]
		print "ConV", newValuesWMConV[2]
		weight = newValuesWMConV[0]
		mean = newValuesWMConV[1]
		conV = newValuesWMConV[2]
		likelihood = newLikelihood
		count += 1
EM(0.0001)