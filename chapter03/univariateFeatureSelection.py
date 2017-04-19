import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df = pd.read_csv('resources/pima-indians-diabetes.csv')
list(df.columns.values)
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]
eval = SelectKBest(score_func=chi2, k=4)
fit = eval.fit(X, y)
fit.scores_
X_NEW = fit.transform(X)
X_NEW.shape