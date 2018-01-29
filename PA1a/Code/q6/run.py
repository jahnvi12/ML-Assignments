#Ridge Regression on multiple samples
import numpy as np
from sklearn.linear_model import Ridge
from scipy import random
from sklearn.metrics import mean_squared_error
random.seed(1)

dataset = np.genfromtxt("../../Dataset/CandC-dataset.csv", delimiter=",")
Y = dataset[:,-1:]
X = np.delete(dataset, 122, 1)
DS1_train = np.genfromtxt("../../Dataset/CandC-train1.csv", delimiter=",")
DS2_train = np.genfromtxt("../../Dataset/CandC-train2.csv", delimiter=",")
DS3_train = np.genfromtxt("../../Dataset/CandC-train3.csv", delimiter=",")
DS4_train = np.genfromtxt("../../Dataset/CandC-train4.csv", delimiter=",")
DS5_train = np.genfromtxt("../../Dataset/CandC-train5.csv", delimiter=",")

DS1_test = np.genfromtxt("../../Dataset/CandC-test1.csv", delimiter=",")
DS2_test = np.genfromtxt("../../Dataset/CandC-test2.csv", delimiter=",")
DS3_test = np.genfromtxt("../../Dataset/CandC-test3.csv", delimiter=",")
DS4_test = np.genfromtxt("../../Dataset/CandC-test4.csv", delimiter=",")
DS5_test = np.genfromtxt("../../Dataset/CandC-test5.csv", delimiter=",")

X_train1 = np.delete(DS1_train, 122, 1)
Y_train1 = DS1_train[:,-1:]
X_test1 = np.delete(DS1_test, 122, 1)
Y_test1 = DS1_test[:,-1:]

X_train2 = np.delete(DS2_train, 122, 1)
Y_train2 = DS2_train[:,-1:]
X_test2 = np.delete(DS2_test, 122, 1)
Y_test2 = DS2_test[:,-1:]

X_train3 = np.delete(DS3_train, 122, 1)
Y_train3 = DS3_train[:,-1:]
X_test3 = np.delete(DS3_test, 122, 1)
Y_test3 = DS3_test[:,-1:]

X_train4 = np.delete(DS4_train, 122, 1)
Y_train4 = DS4_train[:,-1:]
X_test4 = np.delete(DS4_test, 122, 1)
Y_test4 = DS4_test[:,-1:]

X_train5 = np.delete(DS5_train, 122, 1)
Y_train5 = DS5_train[:,-1:]
X_test5 = np.delete(DS5_test, 122, 1)
Y_test5 = DS5_test[:,-1:]

fout=open("result.txt","w+")
alpha = [0.01, 1, 100]
for a in alpha:
	rss = 0.0
	clf = Ridge(alpha=a)
	clf.fit(X_train1, Y_train1)
	np.savetxt("coeffs_"+str(a)+"_1.csv", clf.coef_, delimiter=",")
	y_pred = clf.predict(X_test1)
	rss += mean_squared_error(Y_test1, y_pred)
	clf.fit(X_train2, Y_train2) 
	np.savetxt("coeffs_"+str(a)+"_2.csv", clf.coef_, delimiter=",")
	y_pred = clf.predict(X_test2)
	rss += mean_squared_error(Y_test2, y_pred)
	clf.fit(X_train3, Y_train3) 
	np.savetxt("coeffs_"+str(a)+"_3.csv", clf.coef_, delimiter=",")
	y_pred = clf.predict(X_test3)
	rss += mean_squared_error(Y_test3, y_pred)
	clf.fit(X_train4, Y_train4) 
	np.savetxt("coeffs_"+str(a)+"_4.csv", clf.coef_, delimiter=",")
	y_pred = clf.predict(X_test4)
	rss += mean_squared_error(Y_test4, y_pred)
	clf.fit(X_train5, Y_train5) 
	np.savetxt("coeffs_"+str(a)+"_5.csv", clf.coef_, delimiter=",")
	y_pred = clf.predict(X_test5)
	rss += mean_squared_error(Y_test5, y_pred)
	rss/=5
	fout.write('Residual Sum of Squares for alpha = '+str(a)+' is: ' + str(rss*len(X_test1))+'\n')