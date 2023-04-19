from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    return res
def plotDecisionBoundary(theta,degree, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))

    # Feature mapping
    X_poly = mapFeature(U, V, degree)
    X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))
    Z = X_poly.dot(theta)

    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))

    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
    axes.legend(labels=['class 1', 'class 0', 'Decision Boundary'])
    return cs
degree=4
train = pd.read_csv('train.csv')
train.shape
le = LabelEncoder()
train['class'] = le.fit_transform(train['class'])#make class with 0s and 1s

y = train['class']
x = train.drop(columns=['class'])
x1t = train.drop(columns=['class','x2'])
x2t = train.drop(columns=['class','x1'])
print('=====================')
test = pd.read_csv('test.csv')
test.shape
le2 = LabelEncoder()
test['class'] = le2.fit_transform(test['class'])#make class with 0s and 1s

Ytest = train['class']
Xtest = train.drop(columns=['class'])
x1test = train.drop(columns=['class','x2'])
x2test = train.drop(columns=['class','x1'])
print('=====================')

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
y = y.ravel()
x = mapFeature(x, x, degree)
logisticRegr.fit(x, y)
intercept = logisticRegr.intercept_
coefs = logisticRegr.coef_
optimum = np.vstack((intercept,coefs.reshape(x.shape[1],1)))
print('=====================')
# Plotting decision boundary

fig, axes = plt.subplots();
axes.set_xlabel('x 1')
axes.set_ylabel('x 2')
plt.scatter(x[x==0],x[x==0],c='r',label='C0')
plt.scatter(x[x==1],x[x==1],c='g',label='C1')
plotDecisionBoundary(optimum, degree, axes)
