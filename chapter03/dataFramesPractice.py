#dataFramesPractice.py
import numpy as np
import pandas as pd

df = pd.read_csv('resources/pima-indians-diabetes.csv')
df.describe()

studentDF=pd.DataFrame(np.array([['John', 'Bob', 'Matt', 'Mary'],['William', 'Hopkins', 'Anderson', 'Kaiser'],[9,8,10,11],[56,64,67,88]]))

studentDF=pd.DataFrame(np.array([['John', 'William', 9, 56],
                                 ['Bob', 'Hopkins', 8, 64],
                                 ['Matt', 'Anderson', np.nan, 67],
                                 ['Mel', 'Parker', np.nan, np.nan],
                                 ['Mary', 'Kaiser', 11, 88]]))
studentDF.columns= ["name","lname","age","score"]