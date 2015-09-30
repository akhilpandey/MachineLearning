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
noOfSpamTraining =  1631
noOfHamsTraining =  2509
totalTrainEmail =  4140
noOfSpamTesting =  182
noOfHamsTesting =  279
totalTestEmail =  461

trainingData = []
testingData = []

## Open the spamdata file  ##
fileSpamData = open("spambase.data", 'r')
for line in fileSpamData :
    TmpGrp = EmailCounter % CONSTANT_K
    line = line.strip('\r\n')
    line = line.split(',')
    if(float(line[CONSTANT_F]) == (0.0)):
    	line[CONSTANT_F] = float(-1.0)
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


actualClassification = []
for email in range(totalTrainEmail):
	actualClassification.append(trainingData[email][CONSTANT_F])

actualClassification_test = []
for email in range(totalTestEmail):
	actualClassification_test.append(testingData[email][CONSTANT_F])


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

#############################################################################################
#############################################################################################

# def initialize():
# 	distribution = []
# 	for email in range(totalTrainEmail):
# 		distribution.append(1.0/float(totalTrainEmail))
# 	return distribution

def updateDistribution(distribution, aErrRate, trainPrediction):
	normalizeFactor = 0.0
	for email in range(totalTrainEmail):
		if (float(trainPrediction[email]) == float(actualClassification[email])):
			distribution[email] *= math.sqrt(aErrRate/(1.0 - aErrRate))
		else:
			distribution[email] *= math.sqrt((1.0 - aErrRate)/aErrRate)
		normalizeFactor += distribution[email]
	for email in range(totalTrainEmail):
		distribution[email] = distribution[email]/normalizeFactor
	return distribution


def calculateConfidence(aErrRate):
	return 0.5*(math.log((1.0 - aErrRate)/aErrRate))

def calculateAErrRate(trainPrediction, distribution):	
	aErrRate = 0.0
	for email in range(totalTrainEmail):
		if(float(trainPrediction[email]) != float(actualClassification[email])):
			aErrRate += 1.0
	return aErrRate/totalTrainEmail

def calculateTestMeans(distribution):
	means = []
	for feature in range(CONSTANT_F):
		means.append(0.0)

	for feature in range(CONSTANT_F):
		for email in range(totalTestEmail):
			means[feature] += zscore_test[email][feature]*distribution[feature]

	for feature in range(CONSTANT_F):
		means[feature] /= totalTestEmail
	return means


def NaiveBayesTest(distribution):
	testPrediction = []
	threshold_test = []
	means = calculateTestMeans(distribution)

	threshAndSpam = []
	for feature in range(CONSTANT_F):
		threshAndSpam.append([1,1,1,1])
	# undrThreshSpm = 1
	# overThreshSpm = 1

	# undrThreshHam = 1
	# overThreshHam = 1
	
	for email in range(totalTestEmail):
		for feature in range(CONSTANT_F):
			testingFeatureVal = zscore_test[email][feature]
			tmp = distribution[feature]
			if(float(actualClassification_test[email]) == 1.0):
				if(testingFeatureVal <= means[feature]):
					threshAndSpam[feature][0] += tmp
				else:
					threshAndSpam[feature][1] += tmp
			else:
				if(testingFeatureVal <= means[feature]):
					threshAndSpam[feature][2] += tmp
				else:
					threshAndSpam[feature][3] += tmp

	for feature in range(CONSTANT_F):
		threshAndSpam[feature][0] /= float(threshAndSpam[feature][0] + threshAndSpam[feature][1])
		threshAndSpam[feature][1] /= float(threshAndSpam[feature][0] + threshAndSpam[feature][1])
		threshAndSpam[feature][2] /= float(threshAndSpam[feature][2] + threshAndSpam[feature][3])
		threshAndSpam[feature][3] /= float(threshAndSpam[feature][2] + threshAndSpam[feature][3])

	for email in range(totalTestEmail):
		numerator = 0.0
		denominator = 0.0
		for feature in range(CONSTANT_F):
			if(zscore_test[email][feature] <= means[feature]):
				numerator += math.log(threshAndSpam[feature][0])
				denominator += math.log(threshAndSpam[feature][2])
			else:
				numerator += math.log(threshAndSpam[feature][1])
				denominator += math.log(threshAndSpam[feature][3])
		tmpPredict = math.log(noOfSpamTesting) - math.log(noOfHamsTesting) + numerator - denominator 
		threshold_test.append(tmpPredict)
		if (tmpPredict >= 0):
			testPrediction.append(1.0)
		else:
			testPrediction.append(-1.0)

	return testPrediction,threshold_test

def calculateMeans(distribution):
	means = []
	for feature in range(CONSTANT_F):
		means.append(0.0)

	for feature in range(CONSTANT_F):
		for email in range(totalTrainEmail):
			means[feature] += zscore[email][feature]*distribution[feature]

	for feature in range(CONSTANT_F):
		means[feature] /= totalTrainEmail
	return means


def NaiveBayes(distribution):
	trainPrediction = []
	threshold = []
	# for email in range(totalTrainEmail):
	# 	trainPrediction.append(1.0)
	means = calculateMeans(distribution)

	threshAndSpam = []
	for feature in range(CONSTANT_F):
		threshAndSpam.append([1,1,1,1])
	# undrThreshSpm = 1
	# overThreshSpm = 1

	# undrThreshHam = 1
	# overThreshHam = 1
	
	for email in range(totalTrainEmail):
		for feature in range(CONSTANT_F):
			trainingFeatureVal = zscore[email][feature]
			tmp = distribution[feature]
			if(float(actualClassification[email]) == 1.0):
				if(trainingFeatureVal <= means[feature]):
					threshAndSpam[feature][0] += tmp
				else:
					threshAndSpam[feature][1] += tmp
			else:
				if(trainingFeatureVal <= means[feature]):
					threshAndSpam[feature][2] += tmp
				else:
					threshAndSpam[feature][3] += tmp

	for feature in range(CONSTANT_F):
		threshAndSpam[feature][0] /= float(threshAndSpam[feature][0] + threshAndSpam[feature][1])
		threshAndSpam[feature][1] /= float(threshAndSpam[feature][0] + threshAndSpam[feature][1])
		threshAndSpam[feature][2] /= float(threshAndSpam[feature][2] + threshAndSpam[feature][3])
		threshAndSpam[feature][3] /= float(threshAndSpam[feature][2] + threshAndSpam[feature][3])

	for email in range(totalTrainEmail):
		numerator = 0.0
		denominator = 0.0
		for feature in range(CONSTANT_F):
			if(zscore[email][feature] <= means[feature]):
				numerator += math.log(threshAndSpam[feature][0])
				denominator += math.log(threshAndSpam[feature][2])
			else:
				numerator += math.log(threshAndSpam[feature][1])
				denominator += math.log(threshAndSpam[feature][3])
		tmpPredict = math.log(noOfSpamTraining) - math.log(noOfHamsTraining) + numerator - denominator 
		threshold.append(tmpPredict)
		if (tmpPredict >= 0):
			trainPrediction.append(1.0)
		else:
			trainPrediction.append(-1.0)

	return trainPrediction,threshold


def predictionAnalysis(distribution, prediction, threshold, dataClass):
	fpr = []
	tpr = []

	listOfEmailThresh = threshold
	oderedThresh = sorted(listOfEmailThresh, reverse = True)
	length = len(oderedThresh)
	for thresh in range(length):
		falsePos = 0.0
		falseNeg = 0.0
		truePos = 0.0
		trueNeg = 0.0
		for email in range(length):
			if (float(listOfEmailThresh[email]) >= float(oderedThresh[thresh])):
				#### Predicted as Spam
				if(float(dataClass[email]) == 1.0):
					## Actually Spam
					truePos += 1.0
				else:
					## Actually Ham
					falsePos += 1.0
			else:
				### Predicted as Ham
				if(float(dataClass[email]) == -1.0):
					## Actually ham
					trueNeg += 1.0
				else:
					## Actually Spam
					falseNeg += 1.0
		### Calculate truePos Rate and falsePos rate
		fPosRate = falsePos/(falsePos + trueNeg)
		tPosRate = truePos/(truePos + falseNeg)
		fpr.append(fPosRate)
		tpr.append(tPosRate)
	# plot(fpr,tpr,"red")

	Error = [1.0,1.0]
	for email in range(len(threshold)):
		ActualValue = float(dataClass[email])
		PredictedValue = float(prediction[email])
		if(ActualValue != PredictedValue):
			Error[0]+=distribution[email]
		else:
			Error[1]+=distribution[email]
	accuracy = Error[1]/len(threshold)
	errorRate = Error[0]/len(threshold)

	# print "Gaussian"
	auc = 0
	for i in range(2,len(fpr)):
		auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
	auc = auc * 0.5
	return auc,fpr,tpr,accuracy,errorRate



def Boosting():
	trainPrediction = []
	for email in range(totalTrainEmail):
		trainPrediction.append(1.0)
	testPrediction = []
	for email in range(totalTestEmail):
		testPrediction.append(1.0)

	NoOfRounds = 1

	newAuc = 0.0

	auc = 0.0
	tpr = []
	fpr = []
	accuracy = 0.0
	errorRate = 0.0

	auc_test = 0.0
	fpr_test = []
	tpr_test = []
	accuracy_test = 0.0
	errorRate_test = 0.0

	# distribution = initialize()
	distribution = []
	for email in range(totalTrainEmail):
		distribution.append(1.0/float(totalTrainEmail))
	trainPrediction,threshold = NaiveBayes(distribution)
	testPrediction,threshold_test = NaiveBayesTest(distribution)
	auc, fpr, tpr, accuracy, errorRate = predictionAnalysis(distribution,trainPrediction,threshold, actualClassification)
	auc_test, fpr_test, tpr_test, accuracy_test, errorRate_test = predictionAnalysis(distribution,testPrediction,threshold_test, actualClassification_test)

	while(True):
		aErrRate = calculateAErrRate(trainPrediction, distribution)
		confidence = calculateConfidence(aErrRate)
		distribution = updateDistribution(distribution, aErrRate, trainPrediction)
		print  NoOfRounds,"\tTraining AUC: ", auc, "\tTesting AUC: ", auc_test
		trainPrediction,threshold = NaiveBayes(distribution)
		testPrediction,threshold_test = NaiveBayesTest(distribution)
		newAuc, fpr, tpr,accuracy, errorRate = predictionAnalysis(distribution,trainPrediction,threshold, actualClassification)
		auc_test, fpr_test, tpr_test, accuracy_test,errorRate_test = predictionAnalysis(distribution,testPrediction,threshold_test, actualClassification_test)
		if((newAuc == auc) & (NoOfRounds > 10)):
			print  NoOfRounds,"\tTraining AUC: ", newAuc, "\tTesting AUC: ", auc_test
			break
		auc = newAuc
		NoOfRounds += 1
	plot(fpr,tpr,"red")
Boosting()
show()