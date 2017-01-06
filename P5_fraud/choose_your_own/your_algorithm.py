#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.

######### Naive Bayes ##############
'''  
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print clf.score(features_test, labels_test)
### draw the decision boundary with the text points overlaid

'''

######### SVM ##############
'''
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel = "rbf", max_iter = 10000000, C = 100000000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" # train 2.55 s

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"  # predict 0.334 s

###
#training time: 3.342 s
#predict time: 0.002 s
#Accuracy: 0.96
##
'''


try:
    prettyPicture(clf, features_test, labels_test)
    print "Accuracy: " + str(accuracy_score(pred,labels_test))
except NameError:
    pass
