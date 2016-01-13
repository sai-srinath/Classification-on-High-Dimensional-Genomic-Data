import csv
import sys
import numpy as np
import math
import random
sys.setrecursionlimit(10000000)
def getTrainingandTestingSplits(dataset):
    random.shuffle(dataset)

    slices = [dataset[i::10] for i in xrange(10)]

    for i in xrange(10):
        testing = slices[i]
        training = [item
                    for s in slices if s is not testing
                    for item in s]
       
        yield training, testing

def calc_entropy(p1,p2):
    if p1==0:
        return -1*p2*math.log(p2,2)
    elif p2==0:
        return -1*p1*math.log(p1,2)
    else:        
        return -1*(p1*math.log(p1,2)+p2*math.log(p2,2)) 
    
def entropy(l,index,split):
    
    l0=l1=u1=u0=0.0
    p1=p2=0
    for i in l:
           
          
        if(float(i[index])<=split):
            
            
            p1+=1
            if i[-1]=='0': 
                l0+=1
            else:
                l1+=1            
        else :
            p2+=2
            if i[-1]=='0':
                u0+=1
            else:
                u1+=1            
               
    
    total=p1+p2
    p1=float(p1)/total
    p2=float(p2)/total
    
    ltotal=l0+l1
    pl1=l0/ltotal
    pl2=l1/ltotal
    
    utotal=u0+u1
    
    if(utotal!=0):
        pu1=u0/utotal
        pu2=u1/utotal
    else:
        pu1=pu2=1
    
    
    return p1*calc_entropy(pl1,pl2)+p2*calc_entropy(pu1,pu2)

  
            
def findColumnMean(l,index):
    no=len(l)
    
    mean=0.0
    for i in l:
        mean+=float(i[index])
            
    mean=mean/no
   
    return mean
     
                            
    
class tree:
    def __init__(self,data):
        
        self.data= data     
        self.splitFeature= None
        self.featureValue = None
        self.label = None
        self.traversedColumns= list()
        self.children=[]
     
    def addChild(self,child):
        self.children.append(child)  
        
             
    #copying the functions to make them member functions of class
    def splitList(self,l,index):
        l1=list()
        l2=list()
        splitval=findColumnMean(l,index)
        for i in l:
            if float(i[index])<splitval:
                l1.append(i)
            else:
                l2.append(i)
        return l1,l2      
        
    def findMinColumn(self,l,skipindexes):
        mini=2.0
        mini_index=0
        for i in range(feature_size-1):
        
            if i not in skipindexes:
                splitval=findColumnMean(l,i)
                val=entropy(l,i,splitval)
                if mini>val:
                    mini=val
                    mini_index=i
        
        return mini_index 
    
    def isHomogenous(self,l): 
        if l:                   
            label=l[0][feature_size-1]
            
            flag=True
            for i in l:
                if i[feature_size-1]!=label:
                    flag=False
                    break
            if flag is True:
                return label
            else:
                return -1                   
 
def buildTree(root):
    if root.isHomogenous(root.data)==-1:
        
        index=root.findMinColumn(root.data,root.traversedColumns)
        mean = findColumnMean(root.data,index)
        root.splitFeature=index
        root.featureValue=mean
        l1,l2=root.splitList(root.data,index)
       
        left_child=tree(l1)
        left_child.traversedColumns.append(index)        
        root.addChild(left_child)   
        
        right_child=tree(l2)   
        right_child.traversedColumns.append(index) 
        root.addChild(right_child)
        buildTree(left_child)
        buildTree(right_child)
    else:
        
        val=root.isHomogenous(root.data)
        leaf=tree(list())
        leaf.label=val
        
        root.addChild(leaf) 
        
prediction_list =[]      
def predict(test,rootnode):
    if rootnode.label is not None:
        
        prediction_list.append(rootnode.label)       
        return rootnode.label
    if rootnode.splitFeature is None:
        
        return predict(test,rootnode.children[0])    
    else:
        
        col_id=rootnode.splitFeature
        value=rootnode.featureValue
        
        if(float(test[col_id])<value):
            
            
            return predict(test,rootnode.children[0])
           
        else:
            
           
            return predict(test,rootnode.children[1])        
        
def getAccuracy(test_data, predictions):
	correct = 0
	for x in range(len(test_data)):
	        
	        
		if test_data[x][-1] == predictions[x]:
		        
			correct += 1
	
	return (correct/float(len(test_data))) * 100.0    

def getPrecision_Recall(predictions,test_data):
    eq=neq=req=0
    for x in range(len(test_data)):
        
        if (float(test_data[x][-1]) == 0 and float(predictions[x]) == 0):
            eq =eq+1
        elif( float(predictions[x]) ==0 and float(test_data[x][-1]) == 1):
            neq +=1
        elif( float(predictions[x]) ==1 and float(test_data[x][-1]) == 0):
            req +=1
          
    return (float(eq)/float((eq+neq))),(float(eq)/float((eq+req)))

def getPrecision_Recall_1(predictions,test_data):
    eq=neq=req=0
    for x in range(len(test_data)):
        
        if (float(test_data[x][-1]) == 1 and float(predictions[x]) == 1):
            eq =eq+1
        elif( float(predictions[x]) ==1 and float(test_data[x][-1]) == 0):
            neq +=1
        elif( float(predictions[x]) ==0 and float(test_data[x][-1]) == 1):
            req +=1
          
    return (float(eq)/float((eq+neq))),(float(eq)/float((eq+req)))
                                                                                                            
################### start of program ##############################  
#path = sys.argv[1] 
path='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/project3_dataset2.txt'
reader = csv.reader(open(path),delimiter="\t")
l=list(reader)

for row in l:
        if row[4] == "Absent":
            row[4] = 0
        elif row[4] == "Present":
            row[4] = 1

sum1=p=p2=r2=f2=r=f=0
for train, test in getTrainingandTestingSplits(l):
    
    sample_size=len(train)
    feature_size=len(train[0])
   
    root=tree(train)
    buildTree(root)
   

    for i in test:
       
        predict(i,root)
    
   
    accuracy = getAccuracy(test, prediction_list)
    
    
    sum1 +=accuracy
    
    precision,recall = getPrecision_Recall(prediction_list,test)
   
    p +=precision
    r +=recall
    f_measure = float(2*precision*recall)/float(precision+recall)
    f +=f_measure
    precision2,recall2 = getPrecision_Recall_1(prediction_list,test)
    p2 +=precision2
    r2 +=recall2
    f_measure2 = float(2*precision2*recall2)/float(precision2+recall2)
    f2+=f_measure2
    
    prediction_list =[]
print('Average Accuracy: {0}%').format(sum1/10)  
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



 