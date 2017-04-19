import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('resources/pima-indians-diabetes.csv')
list(df.columns.values)
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]
X.shape
rfc = RandomForestClassifier(n_estimators=100).fit(X, y)
model = SelectFromModel(rfc, prefit=True)
X_NEW = model.transform(X)
X_NEW







