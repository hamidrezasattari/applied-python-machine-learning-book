#imputeMissingData.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer



studentDF=pd.DataFrame(np.array([['John', 'William', 9, 56],
                                 ['Bob', 'Hopkins', 8, 64],
                                 ['Allen', 'Poly', 8, 56],
                                 ['Matt', 'Anderson', np.NaN , 67],
                                 ['Mel', 'Parker', np.NaN  , np.NaN ],
                                 ['Mary', 'Kaiser', 11, 88]]))
studentDF.columns= ["name","lname","age","score"]


imp = Imputer(missing_values="NaN",strategy="mean", axis=0)
imp.fit(studentDF[["age","score"]])
studentImputedDF=imp.transform(studentDF[["age","score"]]);
studentImputedDF