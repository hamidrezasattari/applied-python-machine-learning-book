#ordinalFeatureEncoder.py
import numpy as np
import pandas as pd

studentDF=pd.DataFrame(np.array([['John', 'William', 19, "A","status1"],
                                 ['Bob', 'Hopkins', 18, "B","status2"],
                                 ['Matt', 'Anderson', 20, "D","status4"],
                                 ['Mel', 'Parker', 21, "C","status3"],
                                 ['Mary', 'Kaiser', 18, "B","status2"]]))
studentDF.columns= ["name","lname","age","grade","status_label"]
studentDF



grade_mapping = {'A': 90,
                'B': 80,
                'C': 60,
                'D': 50}

studentDF['grade_1'] = studentDF['grade'].map(grade_mapping)
studentDF
