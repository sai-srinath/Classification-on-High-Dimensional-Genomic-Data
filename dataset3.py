import csv
import sys
import numpy as np
import math
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import PCA

def rotateColumn(a,index):
    tmp=a[0][index]
    for i in range(a.shape[0]-1):
        a[i][index]=a[i+1][index]
    a[a.shape[0]-1][index]=tmp
    return a
    


    
#path1 = sys.argv[1]   
path1='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/dataset3/train.txt'        
reader = csv.reader(open(path1),delimiter=" ")
l=list(reader)

train_mat=np.array((l))
#print train_mat.shape
train_mat=np.transpose(train_mat)
#print train_mat

#print np.var(train_mat,axis=1)
#print train_mat.shape


orig_training=train_mat

#split into train and test
test_mat=train_mat[27:]
train_mat=train_mat[:27]
#print train_mat.shape,test_mat.shape

#read the labels
#path2 = sys.argv[2]
path2='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/dataset3/train_truth.txt' 
reader = csv.reader(open(path2),delimiter=" ")
l2=list(reader)
train_label=np.array((l2))
#appending label column
orig_training=np.hstack((orig_training,train_label))
print "orig training is now",orig_training.shape
#split orig_training into samples with label 0 and label 1
classA=np.array([]).reshape(0,orig_training.shape[1])
classB=np.array([]).reshape(0,orig_training.shape[1])
for i in range(orig_training.shape[0]):
    if orig_training[i][orig_training.shape[1]-1]=='0':
        classA=np.vstack((classA,orig_training[i])) 
    elif orig_training[i][orig_training.shape[1]-1]=='1':  
        classB=np.vstack((classB,orig_training[i]))    
    else:
        print "you're doing some mistake"
#classA=np.delete(classA,classA.shape[1]-1,1)
#classB=np.delete(classB,classB.shape[1]-1,1)
print "classA,classB shapes are:",classA.shape,classB.shape 
#now we have to interpolate classA and classB 
#expanding classA
newclassA=np.array([]).reshape(0,classA.shape[1])
newclassB=np.array([]).reshape(0,classB.shape[1])




for i in range(50):#(classA.shape[1]-6500):
    tmp1=rotateColumn(classA,i)
    newclassA=np.vstack((newclassA,tmp1))
    tmp2=rotateColumn(classB,i)
    newclassB=np.vstack((newclassB,tmp2))                                
print "we have now inflated the training data"
print "newclass A and B shapes are",newclassA.shape,newclassB.shape

train_label=train_label.ravel()
#merging both the classes
finalarray=np.vstack((newclassA,newclassB))
print "finalarray shape is:",finalarray.shape
#shuffling the array
np.random.shuffle(finalarray)

sample_size=finalarray.shape[0]
#train_size=0.8*sample_size
#test_size=0.2*sample_size

#test_label=finalarray[int(train_size):,finalarray.shape[1]-1]
#train_label=finalarray[:int(train_size),finalarray.shape[1]-1]
#print test_label.shape,train_label.shape
labels=finalarray[:,finalarray.shape[1]-1]
print "labels shape is",labels.shape
finalarray=np.delete(finalarray,finalarray.shape[1]-1,1)

#train_mat=finalarray[:int(train_size)]
#test_mat=finalarray[int(train_size):]
#print test_mat.shape,train_mat.shape

#reading test data
#path3 = sys.argv[3]
path3='/Users/saisrinath/Projects/Canopy_Projects/Project3-Classification/dataset3/test.txt' 
reader = csv.reader(open(path3),delimiter=" ")
ltest=list(reader)
test=np.array((ltest))
test=np.transpose(test)


#use SVM
clf=svm.SVC()
clf.fit(finalarray,labels)
print "Predicted values for test are:"
print clf.predict(test)



