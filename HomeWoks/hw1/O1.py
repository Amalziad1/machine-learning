import csv

from numpy import mean
from numpy import std
from numpy.random import randn


HW1=[]
HW2=[]
Midterm=[]
Project=[]
Final=[]

ava=0
av1=0
av2=0
av3=0
av5=0



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

for i in range(0,len(HW1)):
 if( HW1[i]==0.0):
    HW1[i]=ava
 if (HW2[i] == 0.0):
     HW2[i] = av1

 if (Midterm[i] == 0.0):
     Midterm[i] = av2
 if (Project[i] == 0.0):
     Project[i] = av3

 if (Final[i] == 0.0):
     Final[i] = av5


print("HW1 :",HW1)
print(" HW2 :",HW2)
print(" Midterm :",Midterm)
print(" Project :",Project)
print("Final :",Final)








