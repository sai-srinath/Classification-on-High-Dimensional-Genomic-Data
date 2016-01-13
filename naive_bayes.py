import random
import csv
import math
import sys
#path = sys.argv[1]
path='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/project3_dataset2.txt'
#print path
reader = csv.reader(open(path),delimiter="\t")
l = list(reader)

def getTrainingandTestingSplits(dataset):
    random.shuffle(dataset)

    slices = [dataset[i::10] for i in xrange(10)]

    for i in xrange(10):
        testing = slices[i]
        training = [item
                    for s in slices if s is not testing
                    for item in s]
        yield training, testing

def mean_stdev(attributes):
    avg = sum(attributes)/float(len(attributes))
    variance = sum([pow(x-avg,2) for x in attributes])/float(len(attributes)-1)
    variance = math.sqrt(variance)
    return avg,variance

def summarizeByClass(dataset):
        dict = {}
        for i in range(len(dataset)):
	   vector = dataset[i]
	   dict.setdefault(vector[-1],list()).append( vector )
	dict1 = {}
	for key, values in dict.iteritems():
	       
	       summaries = [mean_stdev(attribute) for attribute in zip(*values)]
	       del summaries[-1]
	       dict1[key] = summaries
	
	return dict1

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			#print mean, stdev
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, test):
	predictions = []
	for i in range(len(test)):
		result = predict(summaries, test[i])
		predictions.append(result)
	return predictions

def getAccuracy(test, predictions):
	correct = 0
	for x in range(len(test)):
	        
		if test[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(test))) * 100.0

def getPrecision_Recall_0(predictions,test):
    eq=neq=req=0
    for x in range(len(test)):
        if test[x][-1] == 0 and predictions[x] == 0:
            eq +=1
        elif predictions[x] ==0 and test[x][-1] == 1:
            neq +=1
        elif predictions[x] ==1 and test[x][-1] == 0:
            req +=1
    
    return (float(eq)/float(eq+neq)),(float(eq)/float(eq+req))
def getPrecision_Recall_1(predictions,test):
    eq=neq=req=0
    for x in range(len(test)):
        if test[x][-1] == 1 and predictions[x] == 1:
            eq +=1
        elif predictions[x] ==1 and test[x][-1] == 0:
            neq +=1
        elif predictions[x] ==0 and test[x][-1] == 1:
            req +=1
    
    return (float(eq)/float(eq+neq)),(float(eq)/float(eq+req))
"""*************************Naive Bayes Script Starts here********************************"""

sum1=p1=p2=r1=r2=f1=f2=0
for train, test in getTrainingandTestingSplits(l):
    for row in l:
        if row[4] == "Absent":
            row[4] = 0
        elif row[4] == "Present":
            row[4] = 1
    
    for i in range(len(train)):
        train[i] = [ float(x) for x in train[i] ]
    for i in range(len(test)):
        test[i] = [ float(x) for x in test[i] ]
    summaries = summarizeByClass(train)
    predictions = getPredictions(summaries, test)
    accuracy = getAccuracy(test, predictions)
    sum1=sum1+accuracy
    precision1,recall1 = getPrecision_Recall_0(predictions,test)
    p1 +=precision1
    r1 +=recall1
    precision2,recall2 = getPrecision_Recall_1(predictions,test)
    p2 +=precision2
    r2 +=recall2
    f_measure1 = float(2*precision1*recall1)/float(precision1+recall1)
    f1+=f_measure1
    f_measure2 = float(2*precision2*recall2)/float(precision2+recall2)
    f2+=f_measure2

print "Average Accuray:",sum1/10
print "****************************"
print "Considering class 0 as positive:"
print "Average Precision",p1/10
print "Average Recall:",r1/10
print "Average f_measure:",f1/10
print "****************************"
print "Considering class 1 as positive:"
print "Average Precision",p2/10
print "Average Recall:",r2/10
print "Average f_measure:",f2/10