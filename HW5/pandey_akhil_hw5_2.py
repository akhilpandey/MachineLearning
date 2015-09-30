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

zFeatureData = []
featureData =  []
for feature in range(CONSTANT_F):
	featureData.append([])
	featureData[feature].append([])
	featureData[feature].append([])
	zFeatureData.append([])
	zFeatureData[feature].append([])
	zFeatureData[feature].append([])
	for email in range(totalTrainEmail):
		if(trainingData[email][CONSTANT_F] == 1.0):
			featureData[feature][1].append(zscore[email][feature])
			zFeatureData[feature][1].append(zscore[email][feature])
		else:
			featureData[feature][0].append(zscore[email][feature])
			zFeatureData[feature][0].append(zscore[email][feature])
	featureData[feature][0].sort()
	featureData[feature][1].sort()
	
emTraining_Mean = []
for feature in range(CONSTANT_F):
	emTraining_Mean.append([])
	for SorH in range(2):
		emTraining_Mean[feature].append([])
		magicNum = int(len(featureData[feature][SorH])/(noOfG - 1))
		k = 0
		# print "magicNum", magicNum, "\tlength of", SorH," : ", len(featureData[feature][SorH])
		for gauss in range(noOfG - 1):
			emTraining_Mean[feature][SorH].append(featureData[feature][SorH][k])
			k += magicNum
		if (k >= len(featureData[feature][SorH])):
			k = len(featureData[feature][SorH]) - 1
		emTraining_Mean[feature][SorH].append(featureData[feature][SorH][k])

# print emTraining_Mean[1][1]
# print emTraining_Mean[55][1]


emTraining_Weight = []
for feature in range(CONSTANT_F):
	emTraining_Weight.append([])
	emTraining_Weight[feature].append([])
	emTraining_Weight[feature].append([])
	for SorH in range(2):
		tempTotal = 0
		for gauss in range(noOfG - 1):
			tempTotal += 1.0/float(noOfG)
			emTraining_Weight[feature][SorH].append(1.0/float(noOfG))
		emTraining_Weight[feature][SorH].append(1.0 - tempTotal)


emTraining_ConV = []
for feature in range(CONSTANT_F):
	emTraining_ConV.append([])
	emTraining_ConV[feature].append([])
	emTraining_ConV[feature].append([])
	for SorH in range(2):
		for gauss in range(noOfG):
			conV = 0.0
			for email in range(len(featureData[feature][SorH])):
				if(trainingData[email][CONSTANT_F] == SorH):
					conV += math.pow(emTraining_Mean[feature][SorH][gauss] - zscore[email][feature], 2)
			conV /= totalTrainEmail
			emTraining_ConV[feature][SorH].append(conV)


### Purpose : 
### Contract: 
### Returns : 
def calculatePhi(data, mean, conV):
	expTerm = -(math.pow((data - mean),2)/(2*conV))
	phi = math.exp(expTerm)/math.sqrt((2*pi)*conV)
	if (phi <= 0.001):
		return phi + 0.001
	return phi

### Purpose : 
### Contract: 
### Returns : 
def CalLikelihood(weight, mean, conV, curFeature, spamORham):
	likelihood = 0.0
	dataLength = len(zFeatureData[curFeature][spamORham])
	for data in range(dataLength):
		term1 = 0.0
		for component in range(noOfG):
			term1 += weight[component] * calculatePhi(zFeatureData[feature][spamORham][data], mean[component], conV[component])
		likelihood += math.log(term1)
	likelihood /= float(dataLength)
	# print likelihood
	return likelihood

### Purpose : 
### Contract: 
### Returns : 
def Initialize(curFeature,spamORham):
	mean = emTraining_Mean[curFeature][spamORham]
	conV = emTraining_ConV[curFeature][spamORham]
	weight = emTraining_Weight[curFeature][spamORham]
	likelihood = CalLikelihood(weight,mean,conV, curFeature,spamORham)
	# print curFeature, mean
	return [likelihood, weight, mean, conV]


### Purpose : 
### Contract: 
### Returns : 
def EStep(weight, mean, conV, curFeature,spamORham):	
	dataLength = len(zFeatureData[curFeature][spamORham])
	gamma = []
	for data in range(dataLength):
		gamma.append([])
		for component in range(noOfG):
			gamma[data].append(0.0)
	nComponent = []
	for component in range(noOfG):
		nComponent.append(0.0)
	for data in range(dataLength):
		total = 0.0
		for component in range(noOfG):
			gamma[data][component] = weight[component] * calculatePhi(zFeatureData[curFeature][spamORham][data], mean[component], conV[component])
			total += gamma[data][component]
		for component in range(noOfG):
			gamma[data][component] /= total
	for component in range(noOfG):
		for data in range(dataLength):
			nComponent[component] += gamma[data][component]
	return [gamma,nComponent]

### Purpose : 
### Contract: 
### Returns : 
def MStep(weight, mean, conV, gamma, nComponent,curFeature,spamORham):
	dataLength = len(zFeatureData[curFeature][spamORham])
	newWeight = []
	newMean = []
	newConV = []
	for component in range(noOfG):
		newWeight.append(0.0)
		newMean.append(0.0)
		newConV.append(0.0)
	for component in range(noOfG):
		newWeight[component] = nComponent[component]/dataLength
	for component in range(noOfG):
		for data in range(dataLength):
			newMean[component] += gamma[data][component]*zFeatureData[curFeature][spamORham][data]
		newMean[component] /= nComponent[component]
	for component in range(noOfG):
		for data in range(dataLength):
			newConV[component] += gamma[data][component]*math.pow((zFeatureData[curFeature][spamORham][data] - newMean[component]),2)
		newConV[component] /= nComponent[component]
		if(newConV[component] <= 0.001):
			newConV[component] += 0.001
	return [newWeight, newMean, newConV]

### Purpose : 
### Contract: 
### Returns : 
def EM(threshold,curFeature, spamORham):
	dataLength = len(zFeatureData[curFeature][spamORham])
	likelihood,weight,mean,conV = Initialize(curFeature,spamORham)
	count = 1
	while('true'):
		gammaANDnComponent = EStep(weight,mean,conV,curFeature,spamORham)
		newValuesWMConV = MStep(weight,mean,conV,gammaANDnComponent[0], gammaANDnComponent[1],curFeature,spamORham)
		newLikelihood = CalLikelihood(newValuesWMConV[0],newValuesWMConV[1],newValuesWMConV[2],curFeature,spamORham)
		if((math.fabs(newLikelihood - likelihood) <= threshold) or (count >= 50) or (newLikelihood > 0.0)):
			# print count, newLikelihood
			# print "newWeight", newValuesWMConV[0]
			# print "newMean", newValuesWMConV[1]
			# print "newConV", newValuesWMConV[2]
			return newValuesWMConV
			break
		# print count, likelihood
		weight = newValuesWMConV[0]
		mean = newValuesWMConV[1]
		conV = newValuesWMConV[2]
		likelihood = newLikelihood
		count += 1

prSpamOnHam = math.log(float(len(zFeatureData[1][1]))/float(len(zFeatureData[1][0])))

featureEMSpam = []
featureEMHams = []
for feature in range(CONSTANT_F):
	featureEMSpam.append(EM(0.01,feature,1))
	featureEMHams.append(EM(0.01,feature,0))

threshold = []
for email in range(totalTestEmail):
	threshold.append(0.0)

for email in range(totalTestEmail):
	prGivenSpam = 0.0
	prGivenHams = 0.0
	for feature in range(CONSTANT_F):
		for gauss in range(noOfG):
			prGivenSpam += math.log(featureEMSpam[feature][0][gauss]*calculatePhi(zscore_test[email][feature], featureEMSpam[feature][1][gauss],featureEMSpam[feature][2][gauss]))
			prGivenHams += math.log(featureEMHams[feature][0][gauss]*calculatePhi(zscore_test[email][feature], featureEMHams[feature][1][gauss],featureEMHams[feature][2][gauss]))
	threshold[email] = prSpamOnHam + prGivenSpam - prGivenHams

# for i in range(totalTestEmail):
# 	print threshold[i]

fpr = []
tpr = []

listOfEmailThresh = threshold
oderedThresh = sorted(listOfEmailThresh, reverse = True)
ErrorRate = []
length = len(oderedThresh)
for thresh in range(length):
	falsePos = 0
	falseNeg = 0
	truePos = 0
	trueNeg = 0
	for email in range(length):
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
	ErrorRate.append([fPosRate, tPosRate])
	fpr.append(fPosRate)
	tpr.append(tPosRate)
plot(fpr,tpr,"red")

print "Gaussian"
auc = 0
for i in range(2,len(fpr)):
	auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
auc = auc * 0.5
print noOfG,auc


sampleOutput = [40,55]
for out in range(2):
	print "###################### Feature\t",sampleOutput[out]," ######################"
	print "Ham Weight:\t", featureEMHams[out][0]
	print "Ham Mean:\t", featureEMHams[out][1]
	print "Ham Variance:\t", featureEMHams[out][2]
	print "Spam Weight:\t", featureEMSpam[out][0]
	print "Spam Mean:\t", featureEMSpam[out][1]
	print "Spam Variance:\t", featureEMSpam[out][2]


show()