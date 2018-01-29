# Classification using regression on indicator for 20D Gaussian distribution
import numpy as np 
from scipy import random
from sklearn import linear_model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

dataset1 = np.genfromtxt("../../Dataset/DS1-train.csv", delimiter=",")
Y_train = dataset1[:,-2:]
X_train = np.delete(dataset1, [21,20], 1)
dataset2 = np.genfromtxt("../../Dataset/DS1-test.csv", delimiter=",")
Y_test = dataset2[:,-2:]
X_test = np.delete(dataset2, [21,20], 1)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
np.savetxt("coeffs.csv", regr.coef_, delimiter=",")

array=[1,0]
testClassifier = [a[0] for a in Y_test]
predicted_Y =  regr.predict(X_test)
predictedClassifier = [array[np.argmax(a)] for a in predicted_Y]

fout=open("result.txt","w+")

fout.write('Accuracy score: ' + str(accuracy_score(testClassifier, predictedClassifier)))
fout.write('\nPrecision score: ' + str(precision_score(testClassifier, predictedClassifier)))
fout.write('\nRecall score: ' +  str(recall_score(testClassifier, predictedClassifier)))
fout.write('\nF measure: ' +  str(f1_score(testClassifier, predictedClassifier)))
