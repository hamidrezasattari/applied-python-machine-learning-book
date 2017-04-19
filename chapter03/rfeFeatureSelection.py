import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('resources/pima-indians-diabetes.csv')
list(df.columns.values)
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]
estimator = ExtraTreesClassifier()
rfe = RFE(estimator, 4, step=5)
fit = rfe.fit(X, y)
fit.n_features_
fit.support_
fit.ranking_