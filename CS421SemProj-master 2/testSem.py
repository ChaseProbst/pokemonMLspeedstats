import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#--------------------code for taking in CSV data and converting it into arrays -------------------------#
#path to the csv
path = ""
#names of the columns in the csv
names = []
#combination of both
dataset = read_csv(path,names=names)
#print(type(dataset))
#print(dataset.head(20))
data = dataset.values

#!!!! change to data[:,0:6]
X = data[:,0:4]
print(type(X))
y = data[:,4]
#print(y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X,y,test_size=0.2,random_state=1,shuffle=True)

#print(X_train, Y_validation)


# Creating list of models
models = []
models.append(('Logistic', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('DTree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto')))
#running the different models and 
results=[]
names=[]
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Accuracy')
pyplot.show()