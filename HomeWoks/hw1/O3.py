import csv

import params as params
from numpy.linalg import inv
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
import warnings
import numpy.linalg as LA
from sklearn.metrics import mean_squared_error

from sklearn.svm._libsvm import predict

warnings.filterwarnings('ignore')

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
np.random.seed(42)

with open("grades.csv","r",newline="") as csvfile:
    re=csv.DictReader(csvfile)
    for i in re:
        HW1.append(float (i["HW1"]))
        HW2.append(float (i["HW2"]))
        Midterm.append(float (i["Midterm"]))
        Project.append(float(i["Project"]))
        Final.append(float(i["Final"]))


        av2 = sum(Midterm) / len(Midterm)

        av5 = sum(Final) / len(Final)

    for i in range(0, len(HW1)):


        if (Midterm[i] == 0.0):
            Midterm[i] = av2

        if (Final[i] == 0.0):
            Final[i] = av5




X = np.array(Midterm)
y = np.array(Final)
X_b = np.c_[np.ones(len(Midterm)), X]

# m, b
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)# Normal Equation
print("Theta best:",theta_best) # slope and y-intercept
y_predict = X_b.dot(theta_best)
print("\n Final  predictions:", y_predict)

d=np.array(mean_squared_error(Final,y_predict))
print("\n   error  ",d)