import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

studentDF=pd.DataFrame(np.array([['John', 'William', 19, "A","status1"],
                                 ['Bob', 'Hopkins', 18, "B","status2"],
                                 ['Matt', 'Anderson', 20, "D","status4"],
                                 ['Mel', 'Parker', 21, "C","status3"],
                                 ['Mary', 'Kaiser', 18, "B","status2"]]))
studentDF.columns= ["name","lname","age","grade","status_label"]
studentDF
labelEncoder = LabelEncoder()
studentDF['status_label_1']=labelEncoder.fit_transform(studentDF['status_label'].values)
studentDF


staus_mapping = {'status1': 1,
                'status2': 2,
                'status3': 3,
                'status4': 4}

studentDF['status_label_2'] = studentDF['status_label'].map(staus_mapping)
studentDF
