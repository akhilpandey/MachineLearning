from pylab import *
from numpy import *
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
import math
import numpy
import random
from random import choice

## An counter to keep track of email ##
EmailCounter = 0
## Number of groupNos constants ##
CONSTANT_K = 10
## Number of feilds ##
CONSTANT_F = 57
totalEmail = 4601
noOfG = 10

trainingData = []
testingData = []

## Open the spamdata file  ##
fileSpamData = open("spambase.data", 'r')
for line in fileSpamData :
    TmpGrp = EmailCounter % CONSTANT_K
    line = line.strip('\r\n')
    line = line.split(',')
    tempfeatures = []
    for i in range(len(line)):
    	tempfeatures.append(float(line[i]))
    if (TmpGrp != 0):
    	trainingData.append(tempfeatures)
    else:
    	testingData.append(tempfeatures)
    EmailCounter += 1
fileSpamData.close()

totalTrainEmail = len(trainingData)
totalTestEmail = len(testingData)


#############################################################################################
############################# Normalizing Training Data: ZScore##############################
#############################################################################################

meanOfFeature = []
varianceOfFeature = []
stdDevOfFeature = []
zscore = []

for feature in range(CONSTANT_F):
	meanOfFeature.append(0.0)
	varianceOfFeature.append(0.0)
	stdDevOfFeature.append(0.0)

for email in range(totalTrainEmail):
	zscore.append([])
	for feature in range(CONSTANT_F):
		zscore[email].append(0.0)			
# ### Calculating Mean ###
for feature in range(CONSTANT_F):
	for email in range(totalTrainEmail):
		meanOfFeature[feature] += float(trainingData[email][feature])
for feature in range(CONSTANT_F):
	meanOfFeature[feature] = float(meanOfFeature[feature])/ float(totalTrainEmail)
# ### Calculating Variances ###
for feature in range(CONSTANT_F):
	for email in range(totalTrainEmail):
		varianceOfFeature[feature] += pow(float(trainingData[email][feature]) - meanOfFeature[feature], 2)
for feature in range(CONSTANT_F):
	varianceOfFeature[feature] = varianceOfFeature[feature]/ float(totalTrainEmail - 1)
# ### Calculating Standard Deviation ###
for  feature in range(CONSTANT_F):
	stdDevOfFeature[feature] = pow(varianceOfFeature[feature], 0.5)
# ### Calculating zscore
for feature in range(CONSTANT_F):
	for email in range(totalTrainEmail):
		zscore[email][feature] = (float(trainingData[email][feature]) - meanOfFeature[feature])/stdDevOfFeature[feature]

#############################################################################################
############################# Normalizing Testing Data: ZScore ##############################
#############################################################################################
meanOfFeature_test = []
varianceOfFeature_test = []
stdDevOfFeature_test = []
zscore_test = []

for feature in range(CONSTANT_F):
	meanOfFeature_test.append(0.0)
	varianceOfFeature_test.append(0.0)
	stdDevOfFeature_test.append(0.0)

for email in range(totalTestEmail):
	zscore_test.append([])
	for feature in range(CONSTANT_F):
		zscore_test[email].append(0.0)			
# ### Calculating Mean ###
for feature in range(CONSTANT_F):
	for email in range(totalTestEmail):
		meanOfFeature_test[feature] += float(testingData[email][feature])
for feature in range(CONSTANT_F):
	meanOfFeature_test[feature] = float(meanOfFeature_test[feature])/ float(totalTestEmail)
# ### Calculating Variances ###
for feature in range(CONSTANT_F):
	for email in range(totalTestEmail):
		varianceOfFeature_test[feature] += pow(float(testingData[email][feature]) - meanOfFeature_test[feature], 2)
for feature in range(CONSTANT_F):
	varianceOfFeature_test[feature] = varianceOfFeature_test[feature]/ float(totalTestEmail - 1)
# ### Calculating Standard Deviation ###
for  feature in range(CONSTANT_F):
	stdDevOfFeature_test[feature] = pow(varianceOfFeature_test[feature], 0.5)
# ### Calculating zscore
for feature in range(CONSTANT_F):
	for email in range(totalTestEmail):
		zscore_test[email][feature] = (float(testingData[email][feature]) - meanOfFeature_test[feature])/stdDevOfFeature_test[feature]

def insertIntoList(spamOrHam, dist, emailIdx, kNearestNeighbors, theK):
	
	for neighbour in range(theK):
		if (dist < kNearestNeighbors[neighbour][1]):
			break
	
	kNearestNeighbors.insert(neighbour, [spamOrHam,dist, emailIdx])
	lastItem = kNearestNeighbors[theK]
	kNearestNeighbors.remove(lastItem)
	
	if(len(kNearestNeighbors) != theK):
		print "Somethng has gone wrong in insertIntoList function"
	return kNearestNeighbors


def findKNearestNeighbors(theEmail, theK):
	kNearestNeighbors = []
	for trainEmail in range(theK):
		distance = 0 
		for feature in range(CONSTANT_F):
			distance += math.pow(float(theEmail[feature] - zscore[trainEmail][feature]),2)
		kNearestNeighbors.append([trainingData[trainEmail][CONSTANT_F], distance, trainEmail])
	kNearestNeighbors.sort()

	for trainEmail in range(theK, totalTrainEmail):
		distance = 0 
		for feature in range(CONSTANT_F):
			distance += math.pow(float(theEmail[feature] - zscore[trainEmail][feature]),2)
		if (distance < kNearestNeighbors[theK - 1][1]):
			kNearestNeighbors = insertIntoList(trainingData[trainEmail][CONSTANT_F],distance,trainEmail, kNearestNeighbors, theK)
	return kNearestNeighbors


def countMajority(nearestNeighbors, theK):
	spamCount = 0
	for neighbour in range(theK):
		spamCount += nearestNeighbors[neighbour][0]
	# print nearestNeighbors
	if (spamCount > int(theK/2)):
		return 1,spamCount
	else:
		return 0,spamCount

prediction = []
for email in range(totalTestEmail):
	prediction.append(0)


def KNearestNeighborsAlgo(theK):
	for testEmail in range(totalTestEmail):
		nearestNeighbors = findKNearestNeighbors(zscore_test[testEmail], theK)
		tmp,count  = countMajority(nearestNeighbors, theK)
		threshold.append(float(count)/float(theK))
		prediction[testEmail] = tmp

for i in range(1,10):
	threshold = []
	KNearestNeighborsAlgo(i)
	listOfEmailThresh = threshold
	oderedThresh = list(set(threshold))
	oderedThresh = sorted(oderedThresh, reverse = True)

	fpr = []
	tpr = []
	for thresh in range(len(oderedThresh)):
		falsePos = 0
		falseNeg = 0
		truePos = 0
		trueNeg = 0
		for email in range(len(threshold)):
			if (float(listOfEmailThresh[email]) >= float(oderedThresh[thresh])):
				#### Predicted as Spam
				if(float(testingData[email][CONSTANT_F]) == float(1)):
					## Actually Spam
					truePos += 1
				else:
					## Actually Ham
					falsePos += 1
			else:
				### Predicted as Ham
				if(float(testingData[email][CONSTANT_F]) == float(0)):
					## Actually ham
					trueNeg += 1
				else:
					## Actually Spam
					falseNeg += 1
		### Calculate TruePos Rate and FalsePos rate
		fPosRate = float(falsePos)/float(falsePos + trueNeg)
		tPosRate = float(truePos)/float(truePos + falseNeg)
		fpr.append(fPosRate)
		tpr.append(tPosRate)
	plot(fpr,tpr,"green")
	auc = 0
	for i in range(2,len(fpr)):
		auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
	auc = auc * 0.5
	print i, auc

	gotItRight = 1.0
	gotItWrong = 1.0
	for email in range(totalTestEmail):
		if(float(prediction[email]) == float(testingData[email][CONSTANT_F])):
			gotItRight += 1.0
		else:
			gotItWrong += 1.0

	accuracy = float(totalTestEmail - gotItWrong)/float(totalTestEmail)
	print "Accuracy",accuracy
	show()


def differentKs():
	fpr = []
	tpr = []
	ErrorRate = []
	for i in range(1,11):
		falsePos = 0
		falseNeg = 0
		truePos = 0
		trueNeg = 0
		KNearestNeighborsAlgo(i)
		for testEmail in range(totalTestEmail):
			if (testingData[testEmail][CONSTANT_F] == 1.0):
				## Actually Spam ##
				if(prediction[testEmail] == 1.0):
					## Predicted Spam ##
					truePos += 1
				else:
					## Predicted Ham ##
					falseNeg += 1
			else:
				## Actually Ham ##
				if(prediction[testEmail] == 1.0):
					## Predicted Hpam ##
					falsePos += 1
				else:
					## Predicted Ham ##
					trueNeg += 1
		fPosRate = float(falsePos)/float(falsePos + trueNeg)
		tPosRate = float(truePos)/float(truePos + falseNeg)
		Error = float(falseNeg + falsePos)/float(totalTestEmail)
		ErrorRate.append(Error)
		fpr.append(fPosRate)
		tpr.append(tPosRate)
		

		gotItRight = 1.0
		gotItWrong = 1.0
		for email in range(totalTestEmail):
			if(float(prediction[email]) == float(testingData[email][CONSTANT_F])):
				gotItRight += 1.0
			else:
				gotItWrong += 1.0

		accuracy = float(totalTestEmail - gotItWrong)/float(totalTestEmail)
		print  i, fPosRate, tPosRate, Error,accuracy

# differentKs()