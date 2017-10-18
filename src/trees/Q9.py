import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree
from sklearn.metrics import accuracy_score

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################


########################## DECISION TREE #################################

#### your code goes here

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)
acc = accuracy_score(labels_test, y_pred)


def submitAccuracies():
    return {"acc": round(acc, 3)}
