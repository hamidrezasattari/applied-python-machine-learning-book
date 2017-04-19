import numpy as np
import pandas as pd
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

df = pd.read_csv('resources/pima-indians-diabetes.csv')
dfValuesArray = df.values
X = dfValuesArray[:,0:8]
y = dfValuesArray[:,8]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.4, random_state=0)
X_train.shape