import csv

import inline as inline
import matplotlib
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot, pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing, svm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
from scipy.stats import pearsonr
import pandas as pd

import pandas
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d



HW1=[]
HW2=[]
Midterm=[]
Project=[]
Final=[]

ava=0

with open("grades.csv","r",newline="") as csvfile:
    re=csv.DictReader(csvfile)
    for i in re:
        HW1.append(float (i["HW1"]))
        HW2.append(float (i["HW2"]))
        Midterm.append(float (i["Midterm"]))
        Project.append(float(i["Project"]))
        Final.append(float(i["Final"]))

        ava = sum(HW1) / len(HW1)
        av1 = sum(HW2) / len(HW2)
        av2 = sum(Midterm) / len(Midterm)
        av3 = sum(Project) / len(Project)
        av5 = sum(Final) / len(Final)

for i in range(0, len(HW1)):
    if (HW1[i] == 0.0):
        HW1[i] = ava
    if (HW2[i] == 0.0):
        HW2[i] = av1

    if (Midterm[i] == 0.0):
        Midterm[i] = av2
    if (Project[i] == 0.0):
        Project[i] = av3

    if (Final[i] == 0.0):
        Final[i] = av5

x=np.array(Midterm).reshape((-1,1))
y=np.array(Final)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# Print the Intercept:
print('intercept:', model.intercept_)

# Print the Slope:
print('slope:', model.coef_)

# Predict a Response and print it:
y_pred = model.predict(x)
print('Predicted response:', y_pred, sep='\n')


d=mean_squared_error(Final,y_pred)
print("\n error  ",d)

