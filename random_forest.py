import csv
import sys
import numpy as np
import math
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import random
def getTrainingandTestingSplits(dataset):
    random.shuffle(dataset)

    slices = [dataset[i::10] for i in xrange(10)]

    for i in xrange(10):
        testing = slices[i]
        training = [item
                    for s in slices if s is not testing
                    for item in s]
        ##print "training",testing
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
            #print "i[index] ",i[index],"lesser than ",split
            
            p1+=1
            if i[-1]=='0': 
                l0+=1
            else:
                l1+=1            
        else :#i[index]> split:
            #print "i[index] ",i[index],"greater than ",split
            p2+=2
            if i[-1]=='0':
                u0+=1
            else:
                u1+=1            
               
    #print "l0,l1,u0,u1,p1,p2 are: ",l0,l1,u0,u1,p1,p2
    total=p1+p2
    p1=float(p1)/total
    p2=float(p2)/total
    
    ltotal=l0+l1
    pl1=l0/ltotal
    pl2=l1/ltotal
    
    utotal=u0+u1
    #print "0????????",utotal,u1,u0
    if(utotal!=0):
        pu1=u0/utotal
        pu2=u1/utotal
    else:
        pu1=pu2=1
    
    #print "p1,p2,calc(pl1,pl2),calc(pu1,pu2)",p1,p2,calc_entropy(pl1,pl2),calc_entropy(pu1,pu2)
    return p1*calc_entropy(pl1,pl2)+p2*calc_entropy(pu1,pu2)

  
            
def findColumnMean(l,index):
    no=len(l)
    #print "no is",no
    mean=0.0
    for i in l:
        mean+=float(i[index])
            
    mean=mean/no
   
    return mean

                            
    
class tree:
    def __init__(self,data):
        #self.parent = parent 
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
        #print "i is",i
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
            #print "label is",label
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
        #print "root.data", root.data
        index=root.findMinColumn(root.data,root.traversedColumns)
        mean = findColumnMean(root.data,index)
        root.splitFeature=index
        root.featureValue=mean
        l1,l2=root.splitList(root.data,index)
        #print "lengths of lists to split:",len(l1),len(l2)
        left_child=tree(l1)
        left_child.traversedColumns.append(index)        
        root.addChild(left_child)   
        
        right_child=tree(l2)   
        right_child.traversedColumns.append(index) 
        root.addChild(right_child)
        
        #print"one cycle of split over"
        #recursing
        buildTree(left_child)
        buildTree(right_child)
    else:
        #print "generating leaf"
        #val stores the label ie 0 or 1
        val=root.isHomogenous(root.data)
        leaf=tree(list())
        leaf.label=val
        #print "added label to leaf,label is:",leaf.label
        root.addChild(leaf) 
        #print "what",root.children[0].label
prediction_list =[]     
def predict(test,rootnode):
    if rootnode.label is not None:
        #print "found the label",rootnode.label
        prediction_list.append(rootnode.label)       
        return rootnode.label
    if rootnode.splitFeature is None:
        #print "at a penultimate node",rootnode.children[0].label
        return predict(test,rootnode.children[0])    
    else:
        #print "traversing tree"
        col_id=rootnode.splitFeature
        value=rootnode.featureValue
        #print "col",col_id,value
        if(float(test[col_id])<value):
            #print "test[id] < value case",test[col_id],value
            
            return predict(test,rootnode.children[0])
           
        else:
            #print "test[id] > value case",test[col_id],value
           
            return predict(test,rootnode.children[1])        
        
def getAccuracy(test, predictions):
	correct = 0
	for x in range(len(test)):
	        
	        
		if test[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(test))) * 100.0  
	
def getPrecision_Recall(predictions,test_data):
    eq=neq=req=0
    for x in range(len(test_data)):
        #print test_data[x][-1],predictions[x]
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
#################### Start of Program ##############################
#path = sys.argv[1]
path='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/project3_dataset1.txt'
reader = csv.reader(open(path),delimiter="\t")
alldata=list(reader)

for row in alldata:
        if row[4] == "Absent":
            row[4] = 0
        elif row[4] == "Present":
            row[4] = 1
#print "train and test sizes are:",len(l),len(test)
sum1=p=p2=r2=f2=r=f=0
for l, test in getTrainingandTestingSplits(alldata):
    feature_size=len(l[0])
    total_size=len(l)
#print total_size
    root=tree(l)
    #forest_size=int(sys.argv[2])
    forest_size=5
    tree_size=total_size/forest_size
    #print "individual chunk size is:",tree_size
    #print tree_size
    chunks=list()
    for i in range(forest_size):
        chunk=list()
        for j in range(tree_size):
            chunk.append(l[(i*tree_size)+j])
        chunks.append(chunk)

    #initializing trees: creating the forest
    predictions=[]  
    forests=list() 
    for i in range(forest_size):
        forests.append(tree(chunks[i]))   
        buildTree(forests[i])
    prediction_matrix=np.zeros((forest_size,len(test)))
    #print prediction_matrix.shape
    for i in range(forest_size): 
        for j in range(len(test)):
            prediction_matrix[i][j]=predict(test[j],forests[i])   
            #prediction
        
    #print prediction_matrix
    quorum_result= mode(prediction_matrix,axis=0)   
    result=quorum_result[0]
    #print "result",result
    #finding accuracy
    true_labels=np.zeros(len(test))
    #print true_labels.shape,result.shape
    for i in range(len(test)-1):
        true_labels[i]=(test[i][len(test[0])-1])
         

    #print len(true_labels),len(result.tolist())
    #print "printing true labels and results",true_labels
    result=result.flatten()
    
    #print "checking if dimensions match",true_labels.shape,result.shape
    accuracy=accuracy_score(true_labels,result)
    #print "accuracy is:",accuracy
    sum1 +=accuracy
    precision,recall = getPrecision_Recall(result,test)
   
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
#print('Average Accuracy: {0}%').format(sum1/5)  
print "Average Accuracy(%):",sum1/10*(100)
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