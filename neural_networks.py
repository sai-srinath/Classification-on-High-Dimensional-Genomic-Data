import numpy as np
from scipy.optimize import minimize
from math import sqrt
import csv
import sys


#path = sys.argv[1]
path='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/project3_dataset1.txt'
reader = csv.reader(open(path),delimiter="\t")
l = list(reader)

def getTrainingandTestingSplits(dataset):
    #random.shuffle(dataset)

    slices = [dataset[i::10] for i in xrange(10)]

    for i in xrange(10):
        testing = slices[i]
        training = [item
                    for s in slices if s is not testing
                    for item in s]
        yield training, testing

def initializeWeights(n_in,n_out):
   
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
    
def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    grad_w1=np.zeros(w1.shape)
    grad_w2=np.zeros(w2.shape)
    
    enc=np.array([0]*2) 
    for i in range(training_data.shape[0]):  
        arr = np.array([0]*2) 
            
        index=training_label[i]
        index=int(index)
        arr[index]=1
        enc=np.vstack((enc,arr))
    enc=enc[1:]
    
    training_data=np.c_[training_data,np.ones(training_data.shape[0])]   
    hlayer=np.dot(training_data,w1.T)
    hlayer=sigmoid(hlayer)
    
    hlayer=np.append(hlayer,np.ones([len(hlayer),1]),1)   
    
    #forward pass for second net
        
    out=np.dot(hlayer,w2.T)
    out=sigmoid(out)
    
    tmp_array=enc*np.log(out) + (1-enc)*np.log(1-out)
        
    obj_val+=np.sum(tmp_array)
        
        
        
    delvalue=out-enc
        
    
    grad_w2=np.dot(delvalue.T,hlayer)
    
        
        
    
    tmp=np.zeros((1,n_hidden+1))
    
    tmp=(1-hlayer)*(hlayer)*tmp
    
    tmp= np.delete(tmp,np.s_[-1:],1)
        
    grad_w1=np.dot(tmp.T,training_data)
    
    
    obj_val+=lambdaval*(np.sum(np.square(w1))+np.sum(np.square(w2)))/2
    grad_w1=(grad_w1+lambdaval*w1)/training_data.shape[0]
    grad_w2=(grad_w2+lambdaval*w2)/training_data.shape[0]
    
    
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    obj_val=-1*obj_val/training_data.shape[0]
    
    return (obj_val,obj_grad)
    
    

def nnPredict(w1,w2,data):
    labels = np.array([]).reshape(0,1)
    for i in range(data.shape[0]):
        ip=np.concatenate((data[i,:],[1]))
        ip=ip.reshape(1,ip.size)
    data=np.c_[data,np.ones(data.shape[0])]
    hlayer=np.dot(data,w1.T)
    hlayer=sigmoid(hlayer)
    hlayer=np.append(hlayer,np.ones([len(hlayer),1]),1)
    
    out=np.dot(hlayer,w2.T)
    out=sigmoid(out)
    
    for i in range(out.shape[0]):
        labels=np.vstack((labels,np.argmax(out[i])))

    return labels
    

def getAccuracy(test, predictions):
	correct = 0
	for x in range(len(test)):
	        
		if test[x] == predictions[x]:
	           correct += 1
	
	return (correct/float(len(test))) * 100.0

def getPrecision(predictions,test_label):
    eq=neq=0
    for x in range(len(test_label)):
        
        if (test_label[x] == 0 and predictions[x]==0):
            eq +=1
        elif(predictions[x] ==0 and test_label[x] == 1):
            neq +=1
        
    
    if (eq+neq)!=0:
        return (float(eq)/float(eq+neq))
    elif (eq+neq)==0:
        return 0
    
         
   
def getRecall(predictions,test_label):
     eq=req=0
     for x in range(len(test_label)):
       
        if (test_label[x] == 0 and predictions[x] ==0):
            eq +=1
        elif(predictions[x] ==1 and test_label[x] == 0):
            req +=1
     #print "req", eq,req
     if (eq+req)!=0:
        return (float(eq)/float(eq+req))
     elif (eq+req)==0:
        return 0



def getPrecision_Recall_1(predictions,test):
    eq=neq=req=0
    for x in range(len(test)):
        if test[x][-1] == 1 and predictions[x] == 1:
            eq +=1
        elif predictions[x] ==1 and test[x][-1] == 0:
            neq +=1
        elif predictions[x] ==0 and test[x][-1] == 1:
            req +=1
    if ((eq+neq)!=0 and (eq+req)!=0):
        return (float(eq)/float(eq+neq)),(float(eq)/float(eq+req))
    elif ((eq+neq)==0 and (eq+req)!=0):
        return 0,(float(eq)/float(eq+req))
    elif ((eq+neq)!=0 and (eq+req)==0):
        return (float(eq)/float(eq+neq)),0

#**************Neural Network Script Starts here********************************

s=p=p2=r2=f2=r=f=0
for train, test in getTrainingandTestingSplits(l):
    
    for row in l:
        if row[4] == "Absent":
            row[4] = 0
        elif row[4] == "Present":
            row[4] = 1
    train_label = []
    test_label = []
    train_label = [r1[-1] for r1 in train]
    test_label = [r3[-1] for r3 in test]
    train_data = np.array(train).astype('float64')
    train_label = np.matrix(train_label).astype('float64')
    train_label = train_label.reshape(len(train),1)

    test_data = np.array(test).astype('float64')
    test_label = np.matrix(test_label).astype('float64').reshape(len(test),1)
    
    n_input = train_data.shape[1]


    #n_hidden = int(sys.argv[2])
    n_hidden=40				   
    n_class = 2				   

# initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden);
    initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
    lambdaval = 0.3;


    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    opts = {'maxiter' : 40}    

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    predicted_label = nnPredict(w1,w2,test_data)
    
    s +=getAccuracy(test_label, predicted_label)
    
    precision= getPrecision(predicted_label,test_label)
    recall= getRecall(predicted_label,test_label)
    p +=precision
    r +=recall
    if ((precision+recall)!=0):
        
        f_measure = float(2*precision*recall)/float(precision+recall)
    else:
        
        f_measure = float(2*precision*recall)
    f +=f_measure
    #print predicted_label
    precision2,recall2 = getPrecision_Recall_1(predicted_label,test_label)
    p2 +=precision2
    r2 +=recall2
    if ((precision2+recall2)!=0):
        
        f_measure2 = float(2*precision2*recall2)/float(precision2+recall2)
    else:
        
        f_measure2 = float(2*precision2*recall2)
    f2 +=f_measure2

print "Average Accuray:",s/10
print "****************************"
print "Considering class 0 as positive:"
print "Average Precision:",p/10
print "Average Recall:",r/10
print "Average f_measure:",f/10
print "****************************"
print "Considering class 1 as positive:"
print "Average Precision",p2/10
print "Average Recall:",r2/10
print "Average f_measure:",f2/10