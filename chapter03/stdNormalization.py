#stdNormalization.py
from sklearn import preprocessing
import numpy as np
X = np.array([[ 5., -6.,  10.],[ 12.,  0.,  10.],[ 0.,  12., -11.]])
X_scaled = preprocessing.scale(X)
X_scaled
X_scaled.mean(axis=0)
X_scaled.std(axis=0)


scaler = preprocessing.StandardScaler().fit(X)
scaler
scaler.mean_
scaler.scale_
scaler.transform(X)
scaler.transform([[-1.,  1., 0.]])

