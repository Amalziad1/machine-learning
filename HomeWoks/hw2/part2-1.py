import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

def scatter(filename):
    train = pd.read_csv(filename)
    train.shape
    y = train['class']
    x = train.drop(columns=['class'])
    x1t = train.drop(columns=['class','x2'])
    x2t = train.drop(columns=['class', 'x1'])
    plt.title(filename)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(*x.values[y == 'C1'].T, s=20, alpha=0.5)
    plt.scatter(*x.values[y == 'C2'].T, s=20, alpha=0.5)
    plt.show()


def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i - j)) * (X2 ** j)))

    return res


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def costFunc(theta, X, y):
    m = y.shape[0]
    z = X.dot(theta)
    h = sigmoid(z)
    term1 = y * np.log(h)
    term2 = (1- y) * np.log(1 - h)
    J = -np.sum(term1 + term2, axis = 0) / m
    return J

train = pd.read_csv('train.csv')
train.shape
y = train['class']
X = train.drop(columns=['class'])
x1t = train.drop(columns=['class','x2'])
x2t = train.drop(columns=['class', 'x1'])
degree = 2
X_poly = mapFeature(X.iloc[:, 0], X.iloc[:, 1], degree)
for i in range(len(y)):
    if(y[i]=='C1'):
        y[i]=0
    elif(y[i]=='C2'):
        y[i]=1
y = train.iloc[:, 2]
initial_theta = np.zeros(X_poly.shape[1]).reshape(X_poly.shape[1], 1)
print(initial_theta)
from scipy.optimize import minimize
res = minimize(costFunc, initial_theta, args=(X_poly, y))
theta = res.x
scatter('train.csv')
print('it will be underfitting')