#Logistic Regression
import numpy as np
from subprocess import call
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import classification_report
call('./../../l1_logreg-0.8.2-i686-pc-linux-gnu/l1_logreg_train -s ../../l1_logreg-0.8.2-i686-pc-linux-gnu/Train_features ../../l1_logreg-0.8.2-i686-pc-linux-gnu/Train_labels 0.01 model',shell=True)
call('../../l1_logreg-0.8.2-i686-pc-linux-gnu/l1_logreg_classify model ../../l1_logreg-0.8.2-i686-pc-linux-gnu/Test_features result',shell=True)
#call(['python', '../../Data_LR(DS2)/data_students/data_processing7.py'])
T_test = np.genfromtxt("../../l1_logreg-0.8.2-i686-pc-linux-gnu/Test_labels", skip_header = 2, delimiter="\n")
y_pred = np.genfromtxt("result", skip_header = 7, delimiter="\n")

print T_test, y_pred
print "Classification Report for l1 logreg (Boyds Group): \n"
print(classification_report(T_test, y_pred))

dataset = np.genfromtxt("../../Dataset/DS2-train.csv", skip_header = 2, delimiter=",")
np.random.shuffle(dataset)
T_train = dataset[:,-1:]
X_train = np.delete(dataset,96,1)
dataset = np.genfromtxt("../../Dataset/DS2-test.csv", skip_header = 2, delimiter=",")
np.random.shuffle(dataset)
T_test = dataset[:,-1:]
X_test = np.delete(dataset,96,1)

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train, T_train)

y_pred = logreg.predict(X_test)

print "Classification Report for 2-class Logistic Regression: \n"
print(classification_report(T_test, y_pred))
