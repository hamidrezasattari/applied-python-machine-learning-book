from sklearn.feature_selection import VarianceThreshold
X = [[1, 0, 11, 10], [10, 15, 11, 10], [21, 25, 11,10], [31, 35, 11,10], [41, 45, 12,10], [51, 55,11,10]]
sel = VarianceThreshold(threshold=(0.2))
sel.fit_transform(X)