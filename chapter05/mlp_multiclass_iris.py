
# coding: utf-8

# In[4]:

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import  (StandardScaler,MinMaxScaler)
from sklearn.model_selection import ( KFold , StratifiedKFold, cross_val_score, GridSearchCV) 


import numpy as np

X,Y = load_iris().data, load_iris().target

mlp = MLPClassifier()
mlp.fit(X, Y)

print( mlp.predict([3.1,  2.5,  8.4,  2.2]))
print( mlp.predict_proba([3.1,  2.5,  8.4,  2.2]) )
print ("sum: %f"%np.sum(mlp.predict_proba([3.1,  2.5,  8.4,  2.2])) )



# In[5]:

parameters = {'hidden_layer_sizes':[10,(4,6),(9,20)], 'activation':['relu','logistic']}
X_train = MinMaxScaler().fit_transform(X)

grid_search = GridSearchCV(MLPClassifier( max_iter=50, alpha=0.00001,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1), parameters, cv=10, n_jobs=4, scoring='accuracy')
#print(grid_search.get_params().keys())

grid_search.fit(X_train,Y)
 
 
print(grid_search.best_params_)
print(100*grid_search.best_score_ )


# In[ ]:

X,Y 


# In[ ]:



