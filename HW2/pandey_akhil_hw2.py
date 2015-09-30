from pylab import *
import math


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
fileSpamData = open("hw1spambasedata.txt", 'r')

## Reading file o Data Sructues ##
for line in fileSpamData :
    TmpGrp = EmailCounter % CONSTANT_K
    tempfeatures = line.split(',')
    GroupNo[TmpGrp].append(tempfeatures)
    EmailCounter += 1

fileSpamData.close()

## GroupNo[x][y][z] means x-th groupNo's y-th email's z-th field
# print GroupNo[0][18][57]
# print len(GroupNo) 		# = 10
# print len(GroupNo[1..9]) 	# = 460
# print len(GroupNo[0]) 	# = 461
# print len(GroupNo[x][y]) 	# = 58



###########################################################################
#################### Bernoulli Random Variable#############################
###########################################################################

## Creating an aggregate details of all groups ##

tmpDetails = []
for grp in range(CONSTANT_K):
	tmpDetails.append([])
	for feature in range(CONSTANT_F):
		tmpDetails[grp].append(0)

tmpSpamDetails = []
for grp in range(CONSTANT_K):
	tmpSpamDetails.append([])
	for feature in range(CONSTANT_F):
		tmpSpamDetails[grp].append(0)

tmpHamDetails = []
for grp in range(CONSTANT_K):
	tmpHamDetails.append([])
	for feature in range(CONSTANT_F):
		tmpHamDetails[grp].append(0)

noOfSpams = []
noOfHams = []

noOfSpamsForTesting = []
noOfHamsForTesting = []

for grp in range(CONSTANT_K):
	noOfHams.append(0.0)
	noOfSpams.append(0.0)
	noOfSpamsForTesting.append(0.0)
	noOfHamsForTesting.append(0.0)

for grp in range(CONSTANT_K):
	grpLength = len(GroupNo[grp])
	for feature in range(CONSTANT_F):
		for email in range(grpLength):
			featureValue = float(GroupNo[grp][email][feature])
			tmpDetails[grp][feature] += featureValue
			if float(GroupNo[grp][email][CONSTANT_F - 1]) == float(0):
				tmpHamDetails[grp][feature] += featureValue
			else: 
				tmpSpamDetails[grp][feature] += featureValue
	noOfSpams[grp] = float(tmpDetails[grp][CONSTANT_F - 1])
	noOfHams[grp] = float(grpLength - noOfSpams[grp])


for testGrp in range(CONSTANT_K):
	for grp in range(CONSTANT_K):
		if(testGrp != grp):
			noOfHamsForTesting[testGrp] += noOfHams[grp]
			noOfSpamsForTesting[testGrp] += noOfSpams[grp]



## Combining tmpDetails in order to create aggregate	##
##  training data for respective OverallDataMeans 			##
OverallDataMeans = []
SpamDataMean = []
HamsDataMean = [] 
for grp in range(CONSTANT_K):
		OverallDataMeans.append([])
		SpamDataMean.append([])
		HamsDataMean.append([])
		for feature in range(CONSTANT_F):
			OverallDataMeans[grp].append(0)
			SpamDataMean[grp].append(0)
			HamsDataMean[grp].append(0)



for i in range(CONSTANT_K):
	for j in range(CONSTANT_K):
		if (i != j):
			for feature in range(CONSTANT_F):
				OverallDataMeans[i][feature] += tmpDetails[j][feature]
				SpamDataMean[i][feature] += tmpSpamDetails[j][feature]
				HamsDataMean[i][feature] += tmpHamDetails[j][feature]



## Storeing the average in OverallDataMeans ##
for grp in range(CONSTANT_K):
	for feature in range(CONSTANT_F):
		Hams = noOfHamsForTesting[grp]
		Spams = noOfSpamsForTesting[grp]
		noEmails = Spams + Hams
		OverallDataMeans[grp][feature] = OverallDataMeans[grp][feature] / noEmails
		SpamDataMean[grp][feature] = SpamDataMean[grp][feature] / Spams
		HamsDataMean[grp][feature] = HamsDataMean[grp][feature] / Hams



## Training the data ##
trainingGroupsFor = []
for grp in range(CONSTANT_K):
	trainingGroupsFor.append([])
	for feature in range(CONSTANT_F):
		trainingGroupsFor[grp].append([1,1,1,1])


for testGrp in range(CONSTANT_K):
	for grp in range(CONSTANT_K):
		if(testGrp != grp):
			for feature in range(CONSTANT_F):
				for email in range(len(GroupNo[grp])):
					trainingFeatureValue = float(GroupNo[grp][email][feature])
					if(float(GroupNo[grp][email][CONSTANT_F - 1]) == float(1)):
						if(trainingFeatureValue <= float(SpamDataMean[testGrp][feature])):
							trainingGroupsFor[testGrp][feature][0] += 1
						if(trainingFeatureValue > float(SpamDataMean[testGrp][feature])):
							trainingGroupsFor[testGrp][feature][1] += 1
					else:
						if(trainingFeatureValue <= float(HamsDataMean[testGrp][feature])):
							trainingGroupsFor[testGrp][feature][2] += 1
						if(trainingFeatureValue > float(HamsDataMean[testGrp][feature])):
							trainingGroupsFor[testGrp][feature][3] += 1


# print trainingGroupsFor[0]

for grp in range(CONSTANT_K):
	Spams = float(noOfSpamsForTesting[grp])
	Hams = float(noOfHamsForTesting[grp])
	for feature in range(CONSTANT_F):
		trainingGroupsFor[grp][feature][0] = float(trainingGroupsFor[grp][feature][0] / (Spams + 2))
		trainingGroupsFor[grp][feature][1] = float(trainingGroupsFor[grp][feature][1] / (Spams + 2))
		trainingGroupsFor[grp][feature][2] = float(trainingGroupsFor[grp][feature][2] / (Hams + 2))
		trainingGroupsFor[grp][feature][3] = float(trainingGroupsFor[grp][feature][3] / (Hams + 2))
		

# print trainingGroupsFor[0]
# print trainingGroupsFor[1]

BernoulliGrp1Thresh = []


SpamPrediction = []

for grp in range(CONSTANT_K):
	SpamPrediction.append([])
	for email in range(len(GroupNo[grp])):
		SpamPrediction[grp].append(0)

## Applying Naive Bayes with Laplace smoothing [Logrimithic] ##
for testGrp in range(CONSTANT_K):
	term1 = math.log((noOfSpams[testGrp] + 1) / (noOfHams[testGrp] + 1))
	for email in range(len(GroupNo[testGrp])):
		term2Nu = 0
		term2De = 0
		for feature in range(CONSTANT_F - 1):
			if(float(GroupNo[testGrp][email][feature]) <= float(OverallDataMeans[testGrp][feature])):
				term2Nu += math.log(trainingGroupsFor[testGrp][feature][0])
				term2De += math.log(trainingGroupsFor[testGrp][feature][2])
			else:
				term2Nu += math.log(trainingGroupsFor[testGrp][feature][1])
				term2De += math.log(trainingGroupsFor[testGrp][feature][3])
		tempThreshold = float((term1 + term2Nu - term2De))
		SpamPrediction[testGrp][email] = int(tempThreshold >= float(0))
		if(float(testGrp) == float(0)):
			BernoulliGrp1Thresh.append(tempThreshold)


GroupErrors =[]
for grp in range(CONSTANT_K):
	GroupErrors.append([0,0,0,0])

for grp in range(CONSTANT_K):
	for email in range(len(GroupNo[grp])):
		ActualValue = float(GroupNo[grp][email][CONSTANT_F - 1])
		PredictedValue = float(SpamPrediction[grp][email])
		if(ActualValue != PredictedValue):
			if((ActualValue == float(0)) & (PredictedValue == float(1))):
				GroupErrors[grp][0] += 1
			else:
				GroupErrors[grp][1] += 1
		elif(ActualValue == float(1)):
			GroupErrors[grp][2] += 1
		elif(ActualValue == float(0)):
			GroupErrors[grp][3] += 1

# print GroupErrors

###########################################################################
#################### Gaussian Random Variable##############################
###########################################################################


OverVariance = []
SpamVariance = []
HamsVariance = []

for grp in range(CONSTANT_K):
	OverVariance.append([])
	SpamVariance.append([])
	HamsVariance.append([])
	for feature in range(CONSTANT_F):
		OverVariance[grp].append(0)
		SpamVariance[grp].append(0)
		HamsVariance[grp].append(0)


for testGrp in range(CONSTANT_K):
	for trainingGrp in range(CONSTANT_K):
		if(testGrp != trainingGrp):
			for feature in range(CONSTANT_F):
				SpamMean = float(SpamDataMean[testGrp][feature])
				HamsMean = float(HamsDataMean[testGrp][feature])
				OverMean = float(OverallDataMeans[testGrp][feature])
				for email in range(len(GroupNo[trainingGrp])):
					currFeatureVal = GroupNo[trainingGrp][email][feature]
					OverVariance[testGrp][feature] += math.pow(float(currFeatureVal) - OverMean,2)
					if(float(GroupNo[trainingGrp][email][CONSTANT_F - 1]) == float(1)):
						SpamVariance[testGrp][feature] += math.pow(float(currFeatureVal) - SpamMean,2)
					else:
						HamsVariance[testGrp][feature] += math.pow(float(currFeatureVal) - HamsMean,2)


for grp in range(CONSTANT_K):
	Hams = noOfHamsForTesting[grp]
	Spams = noOfSpamsForTesting[grp]
	noEmails = Spams + Hams
	for feature in range(CONSTANT_F):
		OverVariance[grp][feature] = float(OverVariance[grp][feature]) / float(noEmails - 1)
		SpamVariance[grp][feature] = float(SpamVariance[grp][feature]) / float(Spams -1)
		HamsVariance[grp][feature] = float(HamsVariance[grp][feature]) / float(Hams - 1)


## Smoothing Varriances ##

for grp in range(CONSTANT_K):
	Hams = float(noOfHamsForTesting[grp])
	Spams = float(noOfSpamsForTesting[grp])
	noEmails = Spams + Hams
	lamda = (noEmails) / (noEmails + 2)
	for feature in range(CONSTANT_F):
		SpamVariance[grp][feature] = (lamda*SpamVariance[grp][feature]) + ((1 - lamda)*OverVariance[grp][feature])
		HamsVariance[grp][feature] = (lamda*HamsVariance[grp][feature]) + ((1 - lamda)*OverVariance[grp][feature])


### Testing data ###
GaussianGrp1Thresh = []

SpamPrediction2 = []

for grp in range(CONSTANT_K):
	SpamPrediction2.append([])
	for email in range(len(GroupNo[grp])):
		SpamPrediction2[grp].append(0)

## Applying Naive Bayes with Laplace smoothing [Logrimithic] with Gaussian ##
for testGrp in range(CONSTANT_K):
	term1 = math.log((noOfSpams[testGrp] + 1) / (noOfHams[testGrp] + 1))
	const = 0.5*(float(math.log(2.0*22.0/7.0)))
	for email in range(len(GroupNo[testGrp])):
		term2Nu = 0
		term2De = 0
		for feature in range(CONSTANT_F - 1):
			currFeatureVal = float(GroupNo[testGrp][email][feature])
			term2Nu += math.log(1) - 0.5*(math.log(SpamVariance[testGrp][feature])) - const - 0.5*math.pow((currFeatureVal - SpamDataMean[testGrp][feature]),2)/float(2*SpamVariance[testGrp][feature])
			term2De += math.log(1) - 0.5*(math.log(HamsVariance[testGrp][feature])) - const - 0.5*math.pow((currFeatureVal - HamsDataMean[testGrp][feature]),2)/float(2*HamsVariance[testGrp][feature])
		tempThreshold = float(term1 + (term2Nu - term2De))
		SpamPrediction2[testGrp][email] = int(tempThreshold >= float(0))
		if(float(testGrp) == float(0)):
			GaussianGrp1Thresh.append(tempThreshold)


GroupErrors2 =[]
for grp in range(CONSTANT_K):
	GroupErrors2.append([0,0,0,0])

for grp in range(CONSTANT_K):
	for email in range(len(GroupNo[grp])):
		ActualValue = float(GroupNo[grp][email][CONSTANT_F - 1])
		PredictedValue = float(SpamPrediction2[grp][email])
		if(ActualValue != PredictedValue):
			if(ActualValue == float(0)):
				GroupErrors2[grp][0] += 1
			else:
				GroupErrors2[grp][1] += 1
		elif(ActualValue == float(1)):
			GroupErrors2[grp][2] += 1
		elif(ActualValue == float(0)):
			GroupErrors2[grp][3] += 1



################################################################################################
########################### distribution via a histogram #######################################
################################################################################################

SpambucketCount = []
HambucketCounts = []

for grp in range(CONSTANT_K):
	SpambucketCount.append([])
	HambucketCounts.append([])
	for feature in range(CONSTANT_F):
		SpambucketCount[grp].append([])
		HambucketCounts[grp].append([])
		for i in range(4):
			SpambucketCount[grp][feature].append(1)
			HambucketCounts[grp][feature].append(1)


for testGrp in range(CONSTANT_K):
	for grp in range(CONSTANT_K):
		if (testGrp != grp):
			for feature in range(CONSTANT_F):
				lowMeanValue =  float(SpamDataMean[testGrp][feature])
				highMeanValue = float(HamsDataMean[testGrp][feature])
				overMeanValue = float(OverallDataMeans[testGrp][feature])
				if (highMeanValue < lowMeanValue):
					temp = lowMeanValue
					lowMeanValue = highMeanValue
					highMeanValue = temp
				for email in range(len(GroupNo[grp])):	
					featureValue = float(GroupNo[grp][email][feature])
					if (float(GroupNo[grp][email][CONSTANT_F - 1]) == float(1)):
						if(featureValue <= lowMeanValue):
							SpambucketCount[testGrp][feature][0] += 1
						elif (featureValue <= overMeanValue):
							SpambucketCount[testGrp][feature][1] += 1
						elif (featureValue <= highMeanValue):
							SpambucketCount[testGrp][feature][2] += 1
						else:
							SpambucketCount[testGrp][feature][3] += 1
					else:
						if(featureValue <= lowMeanValue):
							HambucketCounts[testGrp][feature][0] += 1
						elif (featureValue <= overMeanValue):
							HambucketCounts[testGrp][feature][1] += 1
						elif (featureValue <= highMeanValue):
							HambucketCounts[testGrp][feature][2] += 1
						else:
							HambucketCounts[testGrp][feature][3] += 1


for grp in range(CONSTANT_K):
	Hams = float(noOfHamsForTesting[grp]) + 4
	Spams = float(noOfSpamsForTesting[grp]) + 4
	for feature in range(CONSTANT_F):
		for i in range(4):
			SpambucketCount[grp][feature][i] = math.log(SpambucketCount[grp][feature][i] / Spams)
			HambucketCounts[grp][feature][i] = math.log(HambucketCounts[grp][feature][i] / Hams)


### Testing data ###
HistogramGrp1Thresh =[]

SpamPrediction3 = []

for grp in range(CONSTANT_K):
	SpamPrediction3.append([])
	for email in range(len(GroupNo[grp])):
		SpamPrediction3[grp].append(0)

for testGrp in range(CONSTANT_K):
	term1 = math.log((noOfSpams[testGrp] + 2) / (noOfHams[testGrp] + 2))
	for email in range(len(GroupNo[testGrp])):
		term2 = 0
		for feature in range(CONSTANT_F - 1):
			lowMeanValue =  float(SpamDataMean[testGrp][feature])
			highMeanValue = float(HamsDataMean[testGrp][feature])
			overMeanValue = float(OverallDataMeans[testGrp][feature])
			if (highMeanValue < lowMeanValue):
					temp = lowMeanValue
					lowMeanValue = highMeanValue
					highMeanValue = temp
			featureValue = float(GroupNo[testGrp][email][feature])
			if(featureValue <= lowMeanValue):
				term2 += SpambucketCount[testGrp][feature][0] - HambucketCounts[testGrp][feature][0]
			elif (featureValue <= overMeanValue):
				term2 += SpambucketCount[testGrp][feature][1] - HambucketCounts[testGrp][feature][1]
			elif (featureValue <= highMeanValue):
				term2 += SpambucketCount[testGrp][feature][2] - HambucketCounts[testGrp][feature][2]
			else:
				term2 += SpambucketCount[testGrp][feature][3] - HambucketCounts[testGrp][feature][3]
		tempThreshold = float((term1 + term2))
		SpamPrediction3[testGrp][email] = int(tempThreshold > float(0))
		if(float(testGrp) == float(0)):
			HistogramGrp1Thresh.append(tempThreshold)



GroupErrors3 = []
for grp in range(CONSTANT_K):
	GroupErrors3.append([0,0,0,0])

for grp in range(CONSTANT_K):
	for email in range(len(GroupNo[grp])):
		ActualValue = float(GroupNo[grp][email][CONSTANT_F - 1])
		PredictedValue = float(SpamPrediction3[grp][email])
		if(ActualValue != PredictedValue):
			if(ActualValue == float(0)):
				GroupErrors3[grp][0] += 1
			else:
				GroupErrors3[grp][1] += 1
		elif(ActualValue == float(1)):
			GroupErrors3[grp][2] += 1
		elif(ActualValue == float(0)):
			GroupErrors3[grp][3] += 1

color = ["yellow","gray", "red"] 
cntr = 0
fpr = []
tpr = []



listOfEmailThresh = BernoulliGrp1Thresh
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
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(1)):
				## Actually Spam
				truePos += 1
			else:
				## Actually Ham
				falsePos += 1
		else:
			### Predicted as Ham
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(0)):
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
plot(fpr,tpr,"yellow")

print "Bernoulli"
auc = 0
for i in range(2,len(fpr)):
	auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
auc = auc * 0.5
print auc

fpr = []
tpr = []

listOfEmailThresh = GaussianGrp1Thresh
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
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(1)):
				## Actually Spam
				truePos += 1
			else:
				## Actually Ham
				falsePos += 1
		else:
			### Predicted as Ham
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(0)):
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
plot(fpr,tpr,"gray")

print "Gaussian"
auc = 0
for i in range(2,len(fpr)):
	auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
auc = auc * 0.5
print auc

fpr = []
tpr = []

listOfEmailThresh = HistogramGrp1Thresh
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
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(1)):
				## Actually Spam
				truePos += 1
			else:
				## Actually Ham
				falsePos += 1
		else:
			### Predicted as Ham
			if(float(GroupNo[0][email][CONSTANT_F -1]) == float(0)):
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


print "Histogram"
auc = 0
for i in range(2,len(fpr)):
	auc += float((fpr[i] - fpr[i-1])*(tpr[i] + tpr[i-1]))
auc = auc * 0.5
print auc


show()
# print GroupErrors
# print GroupErrors2
# print GroupErrors3