import csv
from numpy.linalg import inv
import inline as inline
from numpy import mean
from numpy import std
from numpy.random import randn
import numpy as np
import seaborn as sns
import warnings
import numpy.linalg as LA
warnings.filterwarnings('ignore')

import seaborn as sns

sns.set(style='darkgrid')
import pandas as pd
import numpy as np
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
        Midterm.append(float (i["Midterm"]))
        Final.append(float(i["Final"]))

        av2 = sum(Midterm) / len(Midterm)
        av5 = sum(Final) / len(Final)

    for i in range(0, len(HW1)):


        if (Midterm[i] == 0.0):
            Midterm[i] = av2

        if (Final[i] == 0.0):
            Final[i] = av5






x = (Midterm)
Y = (Final)



def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))

theta = np.zeros(2)
X = np.column_stack((np.ones(len(x)), x))
y = Y
cost = compute_cost(X, y, theta)
def gradient_descent_multi(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient

    return theta

iterations = 33
alpha = 0.1


theta = gradient_descent_multi(X, y, theta, alpha, iterations)
predictions = X.dot(theta)
print(predictions)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost', cost)