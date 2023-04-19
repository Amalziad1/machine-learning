import csv
#part one without linear boundry === MISSING LINEAR BOUNDRY
import pandas as pd

def read_file(filename):
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        for i in reader:
            dataset.append(i)
    return dataset


def string_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):

            dataset[i][j] = float(dataset[i][j])
    return dataset

def min_max(dataset):
    minmax = []
    for i in range(len(dataset[0])):
            col_val = [j[i] for j in dataset]
            min_ = min(col_val)
            max_ = max(col_val)
            minmax.append([min_, max_])


    return minmax

def normalization(dataset,minmax):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            n = dataset[i][j] - minmax[j][0]
            d = minmax[j][1] - minmax[j][0]
            dataset[i][j] = n/d
    return dataset

def accuracy_check(pred,actual):
    c = 0
    for i in range(len(actual)):
        if(pred[i]==actual[i]):
            c+=1
    acc = (c/len(actual))*100
    return acc
import numpy as np
import math

def prediction(row,parameters):
    hypothesis = parameters[0]
    for i in range(len(row)-2):
        hypothesis+=row[i]*parameters[i+1]
    return 1 / (1 + math.exp(-hypothesis))
def cost_function(x,parameters):
    cost = 0
    for row in x:
        pred = prediction(row,parameters)
        y = row[-1]
        cost+= -(y*np.log(pred))+(-(1-y)*np.log(1-pred))
    avg_cost = cost/len(x)
    return avg_cost


def gradient_descent(x, epochs, alpha):
    parameters = [0] * len(x[0])
    cost_history = []
    n = len(x)

    for i in range(epochs):
        for row in x:
            pred = prediction(row, parameters)
            # for theta 0 partial derivative is different
            parameters[0] = parameters[0] - alpha * (pred - row[-1])
            for j in range(len(row) - 1):
                parameters[j + 1] = parameters[j + 1] - alpha * (pred - row[-1]) * row[j]
        cost_history.append(cost_function(x, parameters))
    return cost_history, parameters


import matplotlib.pyplot as plt


def algorithm(train_data, test_data):
    epochs = 1000
    alpha = 0.001
    cost_history, parameters = gradient_descent(train_data, epochs, alpha)
    predictions = []

    for i in test_data:
        pred = prediction(i, parameters)
        predictions.append(round(pred))
    y_actual = [i[-1] for i in test_data]
    accuracy = accuracy_check(predictions, y_actual)

    iterations = [i for i in range(1, epochs + 1)]
    plt.plot(iterations, cost_history)
    plt.show()
    return accuracy,parameters
def combine():
    train = read_file('train1.csv')
    del train[0]
    train = string_to_float(train)
    test = read_file('test1.csv')
    del test[0]
    test = string_to_float(test)
    minmax = min_max(train)
    train = normalization(train,minmax)
    n=len(train)
    accuracy,parameters = algorithm(train,test)
    scatterplot('train.csv')
    scatterplot('test.csv')
    print('the data of both files is underfitting')
    print('because linear boundary is too simple to separate the 2 classes')
    print("Accuracy =",accuracy)



def scatterplot(filename):
    train = pd.read_csv(filename)
    train.shape
    y = train['class']
    x = train.drop(columns=['class'])
    x1t = train.drop(columns=['class','x2'])
    x2t = train.drop(columns=['class', 'x1'])
    x1min, x1max = min(x1t.values), max(x1t.values)
    x2min, x2max = min(x2t.values), max(x2t.values)
    x1 = np.array([x1min, x1max])
    a1,a2=coeff(x1t,x2t)
    # Coefficients - a1, a2
    x2 = a1 * x1 + a2
    x1train = train.drop(columns=['class', 'x2'])
    x2train = train.drop(columns=['class', 'x1'])
    c1 = train['class'] == 'C1'
    c2 = train['class'] == 'C2'
    plt.title(filename)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.plot(x1, x2, 'k', lw=1, ls='--')
    plt.scatter(*x.values[y == 'C1'].T, s=20, alpha=0.5)
    plt.scatter(*x.values[y == 'C2'].T, s=20, alpha=0.5)
    plt.xlim(x1min, x1max)
    plt.ylim(x2min, x2max)

    plt.show()

def coeff(x1t,x2t):
    xy=x1t.values*x2t.values
    x_square=pow(x1t.values,2)
    y_square=pow(x2t.values,2)
    n=len(x1t)
    a1=((n*sum(xy))-(sum(x1t.values)*sum(x2t.values)))/((n*sum(x_square))-pow(sum(x1t.values),2))
    a2=((sum(x_square)*sum(x2t.values))-(sum(x1t.values)*sum(xy)))/((n*sum(x_square))-pow(sum(x1t.values),2))
    return a1,a2


combine()

