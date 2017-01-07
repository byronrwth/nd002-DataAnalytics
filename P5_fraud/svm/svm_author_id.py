#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score
#clf = svm.SVC(kernel = "rbf", max_iter = 10000000, C = 100000000)

clf = svm.SVC(kernel = 'rbf', C=10000.0)


#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" # train 2.55 s

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"  # predict 0.334 s


pred = clf.predict(features_test)

print accuracy_score(pred, labels_test)

print len(pred), sum(pred)
#########################################################


