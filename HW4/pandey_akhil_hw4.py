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
    line = line.strip('\r\n')
    line = line.split(',')
    tempfeatures = []
    for i in range(len(line)):
    	tempfeatures.append(float(line[i]))
    GroupNo[TmpGrp].append(tempfeatures)
    EmailCounter += 1

fileSpamData.close()


for grp in range(CONSTANT_K):
	for email in range(len(GroupNo[grp])):
		if (float(GroupNo[grp][email][CONSTANT_F - 1]) == float(0.0)):
			GroupNo[grp][email][CONSTANT_F - 1] = float(-1.0)

testingData = GroupNo[0]

trainingData = []

for trainGrp in range(1,10):
	trainingData += GroupNo[trainGrp]


totalEmail = len(trainingData)
totalTestEmail =  len(testingData)

for email in range(totalEmail):
	trainingData[email].insert(CONSTANT_F, float(1.0/float(totalEmail)))

for email in range(totalTestEmail):
	testingData[email].insert(CONSTANT_F, float(1.0/float(totalTestEmail)))

# print len(trainingData)
# print "Di",trainingData[1][CONSTANT_F]
# print "Di",trainingData[1][CONSTANT_F - 1]
# print "Di",trainingData[4001][CONSTANT_F]
# print "Di",trainingData[4001][CONSTANT_F - 1]


featureThresholds = []

for feature in range(CONSTANT_F):
	featureThresholds.append([])


for email in range(totalEmail):
	for feature in range(CONSTANT_F):
		fval = trainingData[email][feature]
		if (fval not in featureThresholds[feature]):
			featureThresholds[feature].append(fval)


for feature in range(CONSTANT_F):
	featureThresholds[feature].sort()


for feature in range(CONSTANT_F):
	for i in range(len(featureThresholds[feature]) - 1):
		featureThresholds[feature][i] = 0.5*(float(featureThresholds[feature][i]) + float(featureThresholds[feature][i+1]))
	featureThresholds[feature].insert(0,featureThresholds[feature][0] - 2.0)
	maxVal = featureThresholds[feature][len(featureThresholds[feature]) - 1]
	featureThresholds[feature][len(featureThresholds[feature]) - 1] = maxVal + 0.5

############################################################################################
################################ Optimal Decision Stumps ###################################
############################################################################################

######### calculating Optimal Decision Stumps ############
def CalcutaleOptimalDS():
	optimalDS = []
	for feature in range(CONSTANT_F + 1):
		optimalDS.append([])	
	#### Calculating error rate wrt first threshold of every feature
	noOfHams = 0.0
	noOfSpam = 0.0
	for email in range(totalEmail):	
		if(trainingData[email][CONSTANT_F - 1] > 0.0):
			noOfSpam += trainingData[email][CONSTANT_F]
		else:
			noOfHams += trainingData[email][CONSTANT_F]
	for feature in range(CONSTANT_F - 1):
		### Adding entries for first and last thresh
		optimalDS[feature].append([featureThresholds[feature][0], abs(0.5 - float(noOfHams)), float(noOfHams),"I"])
		optimalDS[feature].append([featureThresholds[feature][len(featureThresholds[feature]) - 1], abs(0.5 - float(noOfSpam)), float(noOfSpam),"I"])
	###### Calculating error rartes 
	for feature in range(0, CONSTANT_F - 1):
		featureThresError = noOfHams
		trainingData.sort(key = lambda x:float(x[feature]))
		prevData = 0.0
		for email in range(totalEmail):
			actual = trainingData[email][CONSTANT_F - 1]
			emailDi = trainingData[email][CONSTANT_F]
			data = trainingData[email][feature]
			if(email != totalEmail - 1):
				currentThresh = 0.5*(trainingData[email][feature] + trainingData[email + 1][feature])
			else:
				currentThresh = 0.5*(trainingData[email][feature] + 1)
			if(actual == 1.0):
				featureThresError += emailDi
			else:
				featureThresError -= emailDi
			if(currentThresh != data):
				optimalDS[feature].append([currentThresh, abs(0.5 - featureThresError), featureThresError])
				prevData = currentThresh
	##### Get Optimal stumps
	maxAssessError  = optimalDS[0][0][1]
	finalOptimalDS = [0,optimalDS[0][0]]

	for feature in range(CONSTANT_F - 1):
		for i in range(len(optimalDS[feature])):
			tmpAssessError = optimalDS[feature][i][1]
			if (tmpAssessError > maxAssessError):
				maxAssessError = tmpAssessError
				finalOptimalDS = [feature,optimalDS[feature][i]]
	return finalOptimalDS
NoOfRounds = 0



fOfX = []
for email in range(totalEmail):
	fOfX.append(0.0)

fOfXTest = []
for email in range(totalTestEmail):
	fOfXTest.append(0.0)

# print CalcutaleOptimalDS()

prevAUC = 0.0

while(True):
	NoOfRounds += 1
	updateFactorDi = 0.5
	getOptimalDS = [] 

	confidence = 0.0

	getOptimalDS = CalcutaleOptimalDS()
	fi = getOptimalDS[0]
	thresh = getOptimalDS[1][0]

	aErrRate = getOptimalDS[1][1]
	err = getOptimalDS[1][2]
	roundErr = err
	
	err = err/(1-err)
	
	confidence = 0.5*math.log(1.0/err)
	
	updateFactorDi = math.sqrt(err)
	normalizeFactor  = 0.0

	####################### Training Data Stuff ###########################
	for email in range(totalEmail):
		actual = float(trainingData[email][CONSTANT_F - 1])
		if(thresh > float(trainingData[email][fi])):
			fOfX[email] -= confidence
			if ( actual == -1.0):
				trainingData[email][CONSTANT_F] *= updateFactorDi
				normalizeFactor += trainingData[email][CONSTANT_F]
			else:
				trainingData[email][CONSTANT_F] /= updateFactorDi
				normalizeFactor += trainingData[email][CONSTANT_F]
		else:
			fOfX[email] += confidence
			if ( actual == -1.0):
				trainingData[email][CONSTANT_F] /= updateFactorDi
				normalizeFactor += trainingData[email][CONSTANT_F]
			else:
				trainingData[email][CONSTANT_F] *= updateFactorDi
				normalizeFactor += trainingData[email][CONSTANT_F]

	for email in range(totalEmail):
		trainingData[email][CONSTANT_F] /= normalizeFactor

	#### Calculate training error ####
	trainingError  = 0.0

	for email in range(totalEmail):
		if (sign(fOfX[email]) != float(trainingData[email][CONSTANT_F - 1])):
			trainingError += 1.0

	trainingError = float(trainingError/totalEmail)


	######################### Testing Data Stuff ###########################

	for email in range(totalTestEmail):
		if (thresh < float(testingData[email][fi])):
			fOfXTest[email] += confidence
		else:
			fOfXTest[email] -= confidence

	#### Calculate testing error ####
	testingError  = 0.0

	for email in range(totalTestEmail):
		if (sign(fOfXTest[email]) != float(testingData[email][CONSTANT_F - 1])):
			testingError += 1.0

	testingError = float(testingError/totalTestEmail)

	#### Calculate AUc ############
	oderedThresh = sorted(fOfXTest, reverse = True)
	ErrorRate = []
	fpr = []
	tpr = []


	for thr in range(totalTestEmail):
		falsePos = 0.0
		falseNeg = 0.0
		truePos = 0.0
		trueNeg = 0.0
		for email in range(totalTestEmail):
			if (fOfXTest[email] >= oderedThresh[thr]):
				#### Predicted as Spam
				if(float(testingData[email][CONSTANT_F - 1]) == 1.0):
					## Actually Spam
					truePos += 1.0
				else:
					## Actually Ham
					falsePos += 1.0
			else:
				### Predicted as Ham
				if(float(testingData[email][CONSTANT_F - 1]) == -1.0):
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

	#### Calculate AUC #####
	auc = 0
	for i in range(2,len(fpr)):
		auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
	auc = auc * 0.5

	if (roundErr > 0.5):
		roundErr = 1 - roundErr

	if(NoOfRounds == 1):
		prevAUC = auc
	else:
		if(prevAUC == auc):
			print  NoOfRounds,"\tfeature: ", fi, "\tThres: ", thresh, "\tRoundError: ", roundErr, "\tTrainError: ", trainingError, "\tTestError: ", testingError, "\tAUC: ", auc
			plot(fpr,tpr,"red")
			break
		else:
			prevAUC = auc
	print  NoOfRounds,"\tfeature: ", fi, "\tThres: ", thresh, "\tRoundError: ", roundErr, "\tTrainError: ", trainingError, "\tTestError: ", testingError, "\tAUC: ", auc

show()