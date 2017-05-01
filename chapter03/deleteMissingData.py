#deleteMissingData.py
import numpy as np
import pandas as pd

studentDF=pd.DataFrame(np.array([['John', 'Bob', 'Matt', 'Mary'],['William', 'Hopkins', 'Anderson', 'Kaiser'],[9,8,10,11],[56,64,67,88]]))

studentDF=pd.DataFrame(np.array([['John', 'William', 9, 56],
                                 ['Bob', 'Hopkins', 8, 64],
                                 ['Matt', 'Anderson', pd.NaT, 67],
                                 ['Mel', 'Parker', pd.NaT, pd.NaT],
                                 ['Mary', 'Kaiser', 11, 88]]))
studentDF.columns= ["name","lname","age","score"]
studentDF
studentDFCleaned = studentDF.dropna()
studentDFCleaned

#removes instance with two missing value
studentDFCleaned_1=studentDF.dropna(thresh=2)
#removes instance age   missing value
studentDFCleaned_2=studentDF.dropna(subset=["age"])

studentDF[(studentDF.age >= 10)]
