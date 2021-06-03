## Scikit learn --> Supervised Learning --> Classification --> DecisionTreeClassifier

import pandas as pd
import numpy as np

# Reads csv file
dataset = pd.read_csv("done.csv")
# print(dataset)

# Removes last row => Filtering
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 5]
# print(x)
# print(y)

from sklearn.preprocessing import LabelEncoder

# Transforming Label params to int values
x = x.apply(LabelEncoder().fit_transform)
# print(x)

from sklearn.tree import DecisionTreeClassifier

# Instance of DTC
regressor = DecisionTreeClassifier()
# Training data given
regressor.fit(x.iloc[:, 1:5], y)
# print(x.iloc[:,1:5], y)

X_in = np.array([1, 1, 0, 0])
y_pred = regressor.predict([X_in])
print(y_pred)

import pickle
with open('model_pickle', 'wb') as f:  # wb=write
    pickle.dump(regressor, f)
with open('model_pickle', 'rb') as f:  # rb=read
    pf = pickle.load(f)
dr = pf.predict([X_in])
print('dr: ', dr)
