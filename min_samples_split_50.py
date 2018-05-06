#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()
from sklearn import tree
clf=tree.DecisionTreeClassifier(min_samples_split=50,criterion="gini")



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf.fit(features_train, labels_train)

pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc_min_samples_split_50=accuracy_score(pred,labels_test)

print(round(acc_min_samples_split_50,3))

#def submitAccuracies():
  #return {"accuracy":round(acc,3),}






#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())
