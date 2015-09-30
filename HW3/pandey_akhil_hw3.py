from pylab import *
import math
import random


## An counter to keep track of email ##
EmailCounter = 0

## Number of groupNos constants ##
CONSTANT_K = 10
## Number of feilds ##
CONSTANT_F = 58

## Initializing List of List ##
GroupNo = []
for i in range(CONSTANT_K):
	GroupNo.append([])

## Open the spamdata file  ##
fileSpamData = open("spambase.data", 'r')

## Reading file o Data Sructues ##
for line in fileSpamData :
    TmpGrp = EmailCounter % CONSTANT_K
    line = line.rstrip()
    tempfeatures = line.split(',')
    for i in range(len(tempfeatures)):
    GroupNo[TmpGrp].append(tempfeatures)
    EmailCounter += 1

fileSpamData.close()


## GroupNo[x][y][z] means x-th groupNo's y-th email's z-th field
# print GroupNo[0][18][57]
# print len(GroupNo) 		# = 10
# print len(GroupNo[1..9]) 	# = 460
# print len(GroupNo[0]) 	# = 461
# print len(GroupNo[x][y]) 	# = 58

totalEmail = 4601

meanOfFeature = []
varianceOfFeature = []
stdDevOfFeature = []

for feature in range(CONSTANT_F):
	meanOfFeature.append(0.0)
	varianceOfFeature.append(0.0)
	stdDevOfFeature.append(0.0)


zscore = []

for grp in range(CONSTANT_K):
	zscore.append([])
	for email in range(len(GroupNo[grp])):
		zscore[grp].append([])
		for feature in range(CONSTANT_F):
			zscore[grp][email].append(0.0)			


### Calculating Mean ###
for grp in range(CONSTANT_K):
	for feature in range(CONSTANT_F):
		for email in range(len(GroupNo[grp])):
			meanOfFeature[feature] += float(GroupNo[grp][email][feature])
### Smoothing means ###
for feature in range(CONSTANT_F):
	meanOfFeature[feature] = float(meanOfFeature[feature])/ float(totalEmail)



### Calculating Variances ###
for grp in range(CONSTANT_K):
	for feature in range(CONSTANT_F):
		for email in range(len(GroupNo[grp])):
			varianceOfFeature[feature] += pow(float(GroupNo[grp][email][feature]) - float(meanOfFeature[feature]), 2)


### Smoothing Variance ###
for feature in range(CONSTANT_F):
	varianceOfFeature[feature] = float(varianceOfFeature[feature])/ float(totalEmail - 1)

### Calculating Standard Deviation ###
for  feature in range(CONSTANT_F):
	stdDevOfFeature[feature] = pow(varianceOfFeature[feature], 0.5)



### Calculating zscore
for grp in range(CONSTANT_K):
	for feature in range(CONSTANT_F):
		for email in range(len(GroupNo[grp])):
			zscore[grp][email][feature] = float(float(GroupNo[grp][email][feature]) - float(meanOfFeature[feature]) )/float(stdDevOfFeature[feature])



for grp in range(CONSTANT_K):
	for email in range(len(GroupNo[grp])):
		zscore[grp][email][CONSTANT_F - 1] = float(GroupNo[grp][email][CONSTANT_F - 1])

testingData = zscore[0]

trainingData = []

for trainGrp in range(1,10):
	trainingData += zscore[trainGrp]


for email in range(len(testingData)):
	testingData[email].insert(0,1.0)
for email in range(len(trainingData)):
	trainingData[email].insert(0,1.0)

random.shuffle(testingData)
random.shuffle(trainingData)


featureWeights = [0.0]*CONSTANT_F
###########################################################################
#################### Linear Regression : Stochastic #######################
###########################################################################

def linearStoch(lamVal):
	SSE = 0.0
	oldRMS = 0.0
	newRMS = 0.0


	passNo = 0
	lamda = float(lamVal)
	while(True):
		passNo += 1
		SSE = 0.0
		gradientDes = 0.0
		newRMS = 0.0
		for email in range(len(trainingData)):
			predictorTrData = 0.0
			for feature in range(CONSTANT_F):
				predictorTrData += float(featureWeights[feature]) * float(trainingData[email][feature])

			SSE += math.pow(predictorTrData - float(trainingData[email][CONSTANT_F]), 2)

			for feature in range(CONSTANT_F):
				gradientDes = (predictorTrData - float(trainingData[email][CONSTANT_F])) * float(trainingData[email][feature])
				featureWeights[feature] -= float(lamda * gradientDes)

		newRMS = math.sqrt(SSE/float(len(trainingData)))
		print passNo, newRMS

		if (math.fabs(newRMS - oldRMS) < (newRMS * 0.001)):
			print "Linear Stochastic Done"
			break
		oldRMS = newRMS

# linearStoch(0.001)
# linearStoch(0.00001)
#### Best # linearStoch #####
# linearStoch(0.0001)



# ###########################################################################
# ###################### Linear Regression : Batch ##########################
# ###########################################################################


def linearBatch(lamVal):
	oldRMS = 0.0
	passNo = 0
	lamda = float(lamVal)

	while(True):
		passNo += 1
		SSE = 0.0
		gradientDes = [0.0]*CONSTANT_F
		predictorTrData = [0.0]*len(trainingData)

		for email in range(len(trainingData)):
			for feature in range(CONSTANT_F):
				predictorTrData[email] += float(featureWeights[feature]) * float(trainingData[email][feature])			
				gradientDes[feature] += (predictorTrData[email] - float(trainingData[email][CONSTANT_F])) * float(trainingData[email][feature])

		for feature in range(CONSTANT_F):		
			featureWeights[feature] -= float(lamda * gradientDes[feature])
			

		# print featureWeights

		for email in range(len(trainingData)):
			for feature in range(feature):
				predictorTrData[email] += float(featureWeights[feature]) * float(trainingData[email][feature])
			SSE += math.pow((predictorTrData[email] - float(trainingData[email][CONSTANT_F])), 2)

		newRMS = math.sqrt(SSE/float(len(trainingData)))
		print passNo, newRMS

		if (math.fabs(newRMS - oldRMS) < newRMS * 0.001):
			print "Linear Batch Done"
			break
		oldRMS = newRMS



# LinearBatch(0.00005)
# LinearBatch(0.00001)
##### Best linearBatch  #####
# linearBatch(0.0001)

###########################################################################
################## Logistic Regression : Stochastic #######################
###########################################################################

def logisticStoch(lamVal):

	passNo = 0
	lamda = float(lamVal)
	oldRMS = 0.0

	while(True):
		passNo += 1
		
		for email in range(len(trainingData)):
			predictorTrData = 0.0
			for feature in range(CONSTANT_F):
				predictorTrData += float(featureWeights[feature]) * float(trainingData[email][feature])

			predictorTrData = 1.0/(1.0 + math.exp(-predictorTrData))

			for feature in range(CONSTANT_F):
				featureWeights[feature] -= lamda * (predictorTrData - trainingData[email][CONSTANT_F]) * float(trainingData[email][feature]) * (predictorTrData) * (1.0 - predictorTrData)

		SSE = 0.0
		for email in range(len(trainingData)):
			predictorTrData = 0.0
			for feature in range(CONSTANT_F):
				predictorTrData += float(featureWeights[feature]) * float(trainingData[email][feature])
			predictorTrData = 1.0/(1.0 + math.exp(-predictorTrData))
			SSE += math.pow(predictorTrData - float(trainingData[email][CONSTANT_F]), 2)

		newRMS = math.sqrt(SSE/float(len(trainingData)))
		print "Logistic Stochastic: ", passNo, newRMS

		if (abs(newRMS - oldRMS) < newRMS * 0.001):
			print "Logistic Stochastic Done"
			break
		oldRMS = newRMS


# logisticStoch(0.01)
# logisticStoch(0.005)
#### Best logisticStoch ####
# logisticStoch(0.05)


###########################################################################
################### Logistiic Regression : Batch ##########################
###########################################################################

def logisticBatch(lamVal):
	
	passNo = 0
	lamda = float(lamVal)
	oldRMS = 0.0

	while(True):
		passNo += 1
		SSE = 0.0
		gradientDes = [0.0]*CONSTANT_F
		for email in range(len(trainingData)):
			predictorTrData = 0.0
			for feature in range(CONSTANT_F):
				predictorTrData += float(featureWeights[feature]) * float(trainingData[email][feature])			
			predictorTrData = 1.0/float(1 + math.exp(-predictorTrData))
			for feature in range(feature):
				gradientDes[feature] += (predictorTrData - float(trainingData[email][CONSTANT_F])) * float(trainingData[email][feature]) * (predictorTrData) * (1.0 -predictorTrData)

		
		# print featureWeights

		for email in range(len(trainingData)):
			predictorTrData = 0.0
			for feature in range(CONSTANT_F):
				predictorTrData += float(featureWeights[feature]) * float(trainingData[email][feature])
			predictorTrData = 1.0/float(1 + math.exp(-predictorTrData))
			SSE += math.pow((predictorTrData - float(trainingData[email][CONSTANT_F])), 2)

		for feature in range(CONSTANT_F):		
			featureWeights[feature] -= float(lamda * gradientDes[feature])

		newRMS = math.sqrt(SSE/float(len(trainingData)))
		print "Logistic Batch: ", passNo, newRMS

		if (math.fabs(newRMS - oldRMS) < newRMS * 0.001):
			print "Logistic Batch Done"
			break
		oldRMS = newRMS


### logisticBatch(0.05)
### logisticBatch(0.005)
### Best logisticBatch ###
# logisticBatch(0.01)


def plotLinearROC(myColor):
	fpr = []
	tpr = []
	length = len(testingData)
	PredictTestData = [0.0]*length

	########### For Linear Regression #########################

	for email in range(length):
		for feature in range(CONSTANT_F):
			PredictTestData[email] += featureWeights[feature] * testingData[email][feature]


	print PredictTestData

	oderedThresh = sorted(PredictTestData, reverse = True)
	ErrorRate = []

	for thresh in range(length):
		falsePos = 0
		falseNeg = 0
		truePos = 0
		trueNeg = 0
		for email in range(length):
			if (PredictTestData[email] >= oderedThresh[thresh]):
				#### Predicted as Spam
				if(float(testingData[email][CONSTANT_F]) == 1.0):
					## Actually Spam
					truePos += 1.0
				else:
					## Actually Ham
					falsePos += 1.0
			else:
				### Predicted as Ham
				if(float(testingData[email][CONSTANT_F]) == 0.0):
					## Actually ham
					trueNeg += 1.0
				else:
					## Actually Spam
					falseNeg += 1.0
		### Calculate TruePos Rate and FalsePos rate
		fPosRate = float(falsePos)/float(falsePos + trueNeg)
		tPosRate = float(truePos)/float(truePos + falseNeg)
		ErrorRate.append([fPosRate, tPosRate])
		fpr.append(fPosRate)
		tpr.append(tPosRate)
	plot(fpr,tpr,myColor)

	auc = 0
	for i in range(2,len(fpr)):
		auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
	auc = auc * 0.5
	print auc

def plotLogicticROC(myColor):
	fpr = []
	tpr = []
	length = len(testingData)
	PredictTestData = [0.0]*length


	for email in range(length):
		for feature in range(CONSTANT_F):
			PredictTestData[email] += featureWeights[feature] * testingData[email][feature]
		PredictTestData[email] = 1.0/float(1 + math.exp(-PredictTestData[email]))

	print PredictTestData

	oderedThresh = sorted(PredictTestData, reverse = True)
	ErrorRate = []

	for thresh in range(length):
		falsePos = 0
		falseNeg = 0
		truePos = 0
		trueNeg = 0
		for email in range(length):
			if (PredictTestData[email] >= oderedThresh[thresh]):
				#### Predicted as Spam
				if(float(testingData[email][CONSTANT_F]) == 1.0):
					## Actually Spam
					truePos += 1.0
				else:
					## Actually Ham
					falsePos += 1.0
			else:
				### Predicted as Ham
				if(float(testingData[email][CONSTANT_F]) == 0.0):
					## Actually ham
					trueNeg += 1.0
				else:
					## Actually Spam
					falseNeg += 1.0
		### Calculate TruePos Rate and FalsePos rate
		fPosRate = float(falsePos)/float(falsePos + trueNeg)
		tPosRate = float(truePos)/float(truePos + falseNeg)
		ErrorRate.append([fPosRate, tPosRate])
		fpr.append(fPosRate)
		tpr.append(tPosRate)
	plot(fpr,tpr,myColor)

	auc = 0
	for i in range(2,len(fpr)):
		auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
	auc = auc * 0.5
	print auc


def Hw3():
	featureWeights = [0.0]*CONSTANT_F
	linearStoch(0.0001)
	plotLinearROC("red")
	featureWeights = [0.0]*CONSTANT_F
	linearBatch(0.0001)
	plotLinearROC("green")
	featureWeights = [0.0]*CONSTANT_F
	logisticStoch(0.05)
	plotLogicticROC("yellow")
	featureWeights = [0.0]*CONSTANT_F
	logisticBatch(0.01)
	plotLogicticROC("blue")
	show()

Hw3()