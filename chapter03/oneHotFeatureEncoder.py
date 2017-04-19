import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


shirtDF=pd.DataFrame(np.array([['red', 1, 19, "loc1"],
                                 ['blue', 1, 28, "loc2"],
                                 ['green', 2, 48, "loc3"],
                                 ['red', 3, 21, "loc1"],
                                 ['red', 4, 45, "loc3"]]))
shirtDF.columns= ['color','size','price' ,'geo_loc']
shirtDF

labelEncoder = LabelEncoder()
shirtDF['color']=labelEncoder.fit_transform(shirtDF['color'].values)
X = shirtDF[['color', 'size', 'price']].values
X
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X_new = oneHotEncoder.fit_transform(X).toarray()
X_new







