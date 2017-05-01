#treeBasedFeatureSelection.py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


df = pd.read_csv('resources/pima-indians-diabetes.csv')
list(df.columns.values)
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_NEW = model.transform(X)
X_NEW



