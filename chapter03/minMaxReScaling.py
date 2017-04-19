from sklearn import preprocessing
import numpy as np

X = np.array([[ 5., -6.,  10.],[ 12.,  0.,  10.],[ 0.,  12., -11.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_minmax

X_test = np.array([[10,-23,14]])
min_max_scaler.transform(X_test)
