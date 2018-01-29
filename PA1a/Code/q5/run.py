import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from scipy import random
random.seed(1)

dataset = np.genfromtxt("../../Dataset/CandC-dataset.csv", delimiter=",")
Y = dataset[:,-1:]
X = np.delete(dataset, 122, 1)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2)
DS2_train1 = np.append(X_train1, Y_train1, 1)
DS2_test1 = np.append(X_test1, Y_test1, 1)
np.savetxt("../../Dataset/CandC-train1.csv", DS2_train1, delimiter=",")
np.savetxt("../../Dataset/CandC-test1.csv", DS2_test1, delimiter=",")

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.2)
DS2_train2 = np.append(X_train2, Y_train2, 1)
DS2_test2 = np.append(X_test2, Y_test2, 1)
np.savetxt("../../Dataset/CandC-train2.csv", DS2_train2, delimiter=",")
np.savetxt("../../Dataset/CandC-test2.csv", DS2_test2, delimiter=",")

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X, Y, test_size=0.2)
DS2_train3 = np.append(X_train3, Y_train3, 1)
DS2_test3 = np.append(X_test3, Y_test3, 1)
np.savetxt("../../Dataset/CandC-train3.csv", DS2_train3, delimiter=",")
np.savetxt("../../Dataset/CandC-test3.csv", DS2_test3, delimiter=",")

X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X, Y, test_size=0.2)
DS2_train4 = np.append(X_train4, Y_train4, 1)
DS2_test4 = np.append(X_test4, Y_test4, 1)
np.savetxt("../../Dataset/CandC-train4.csv", DS2_train4, delimiter=",")
np.savetxt("../../Dataset/CandC-test4.csv", DS2_test4, delimiter=",")

X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X, Y, test_size=0.2)
DS2_train5 = np.append(X_train5, Y_train5, 1)
DS2_test5 = np.append(X_test5, Y_test5, 1)
np.savetxt("../../Dataset/CandC-train5.csv", DS2_train5, delimiter=",")
np.savetxt("../../Dataset/CandC-test5.csv", DS2_test5, delimiter=",")
rss = 0.0
regr = linear_model.LinearRegression()
regr.fit(X_train1, Y_train1)
y_pred = regr.predict(X_test1)
rss += mean_squared_error(Y_test1, y_pred)
np.savetxt("coeffs_1.csv", regr.coef_, delimiter=",")

regr.fit(X_train2, Y_train2)
y_pred = regr.predict(X_test2)
rss += mean_squared_error(Y_test2, y_pred)
np.savetxt("coeffs_2.csv", regr.coef_, delimiter=",")

regr.fit(X_train3, Y_train3)
y_pred = regr.predict(X_test3)
rss += mean_squared_error(Y_test3, y_pred)
np.savetxt("coeffs_3.csv", regr.coef_, delimiter=",")

regr.fit(X_train4, Y_train4)
y_pred = regr.predict(X_test4)
rss += mean_squared_error(Y_test4, y_pred)
np.savetxt("coeffs_4.csv", regr.coef_, delimiter=",")

regr.fit(X_train5, Y_train5)
y_pred = regr.predict(X_test5)
rss += mean_squared_error(Y_test5, y_pred)
np.savetxt("coeffs_5.csv", regr.coef_, delimiter=",")

rss/=5

fout=open("result.txt","w+")
fout.write('Residual Sum of Squares: ' + str(rss*len(X_test1)))