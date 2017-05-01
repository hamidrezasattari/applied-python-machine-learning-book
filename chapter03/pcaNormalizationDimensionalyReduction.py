#pcaNormalizationDimensionalyReduction.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('resources/pima-indians-diabetes.csv')
list(df.columns.values)
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]

rfc = RandomForestClassifier(n_estimators=100).fit(X, y)
model = SelectFromModel(rfc, prefit=True)
X_SEL = model.transform(X)

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])

pipeline.fit(X_SEL);
X_COMP=pipeline.transform(X_SEL);
X_COMP

plt.figure()

reds = y == 0
blues = y == 1
plt.subplot(2, 2, 2, aspect='equal')
plt.plot(X_COMP[reds, 0], X_COMP[reds, 1], "ro")
plt.plot(X_COMP[blues, 0], X_COMP[blues, 1], "bo")
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")
plt.show()