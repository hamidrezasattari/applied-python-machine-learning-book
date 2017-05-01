#getDummisFeatureEncoder.py
import numpy as np
import pandas as pd


shirtDF=pd.DataFrame(np.array([['red', 1, 19, "loc1"],
                                 ['blue', 1, 28, "loc2"],
                                 ['green', 2, 48, "loc3"],
                                 ['red', 3, 21, "loc1"],
                                 ['red', 4, 45, "loc3"]]))
shirtDF.columns= ['c','s','p' ,'gl']
shirtDF


shirtDFDummis = pd.get_dummies(shirtDF[['c','gl']])

shirtDFDummis




