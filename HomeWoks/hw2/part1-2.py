import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


train = pd.read_csv('train.csv')
train.shape
le = LabelEncoder()
train['class'] = le.fit_transform(train['class'])#make class with 0s and 1s

y = train['class']
x = train.drop(columns=['class'])
x1t = train.drop(columns=['class','x2'])
x2t = train.drop(columns=['class','x1'])

log_reg = LogisticRegression()
log_reg.fit(x,y)

# Intercept - a0
a0=log_reg.intercept_
# Coefficients - a1, a2 respectively
coef=log_reg.coef_
a1 = log_reg.coef_[0][0]
a2 = log_reg.coef_[0][1]
# Predicting labels for the given dataset
label_predictions = log_reg.predict(x)

x1min, x1max=min(x1t.values),max(x1t.values)
x2min, x2max=min(x2t.values),max(x2t.values)
x1 = np.array([x1min, x1max])

# Calculate the intercept and gradient of the decision boundary.
c = -a0/a2
m = -a1/a2
x2 = m*x1 + c
# Plotting the decision boundary for training data
plt.title('Logistic classification for training data', fontweight='bold', fontsize=16)
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x1, x2, 'k', lw=1, ls='--')
plt.scatter(*x.values[y==0].T, s=8, alpha=0.5)
plt.scatter(*x.values[y==1].T, s=8, alpha=0.5)
plt.xlim(x1min, x1max)
plt.ylim(x2min, x2max)
scatter0 = plt.scatter(train['x1'], train['x2'], c=train['class'], cmap='viridis')
plt.legend(*scatter0.legend_elements(),
           loc = 'best',
           title = 'Class')
plt.show()
acc=accuracy_score(y, label_predictions)
print('Accuracy for training data= ',acc)
# ================================================================================
test = pd.read_csv('test.csv')
test.shape
le_t = LabelEncoder()
test['class'] = le_t.fit_transform(test['class'])#make class with 0s and 1s
ytest=test['class']
xtest=test.drop(columns=['class'])
x1test = test.drop(columns=['class','x2'])
x2test = test.drop(columns=['class','x1'])



log_reg_test = LogisticRegression()
log_reg_test.fit(xtest,ytest)

labeltest_predictions = log_reg_test.predict(xtest)

# Intercept - a0
a0t=log_reg_test.intercept_
# Coefficients - a1, a2 respectively
coeft=log_reg_test.coef_
a1t = log_reg_test.coef_[0][0]
a2t = log_reg_test.coef_[0][1]

x1mint, x1maxt=min(x1test.values),max(x1test.values)
x2mint, x2maxt=min(x2test.values),max(x2test.values)
x1testing = np.array([x1mint, x1maxt])

ct = -a0t/a2t
mt = -a1t/a2t
x2testing = mt*x1testing + ct

plt.title('Logistic classification for testing data', fontweight='bold', fontsize=16)
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x1testing, x2testing, 'k', lw=1, ls='--')
plt.scatter(*xtest.values[ytest==0].T, s=8, alpha=0.5)
plt.scatter(*xtest.values[ytest==1].T, s=8, alpha=0.5)
plt.xlim(x1mint, x1maxt)
plt.ylim(x2mint, x2maxt)
scatter = plt.scatter(test['x1'], test['x2'], c=test['class'], cmap='viridis')
plt.legend(*scatter.legend_elements(),
           loc = 'best',
           title = 'Class')
plt.show()

accu=accuracy_score(ytest, labeltest_predictions)

print('Accuracy for tesring data= ',accu)

