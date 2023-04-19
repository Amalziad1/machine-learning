import csv
import inline as inline
import matplotlib
from numpy import mean
from numpy import std
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from scipy.stats import pearsonr




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


        ava=sum(HW1) / len(HW1)
        av1 = sum(HW2) / len(HW2)
        av2= sum(Midterm) / len(Midterm)
        av3= sum(Project) / len(Project)
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

# calculate Pearson's correlation

corr1, _ = pearsonr(HW1, Final)
corr2, _ = pearsonr(HW2, Final)
corr3, _ = pearsonr(Midterm, Final)
corr4, _ = pearsonr(Project, Final)
print('HW1 with Final  correlation: %.3f' % corr1)
print('HW2 with final  correlation: %.3f' % corr2)
print('mid with final  correlation: %.3f' % corr3)
print('project with final correlation: %.3f' % corr4)
pyplot.scatter(Midterm, Final)
pyplot.show()
