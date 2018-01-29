# Generate 2000 data points for 20D Gaussian distribution
import numpy as np 
import sys
from scipy import random
from sklearn.model_selection import train_test_split

matrixSize = 20
numDataPts = 2000
delta = 0.7
random.seed(9001)
while 1:
	A = random.rand(matrixSize,matrixSize)
	cov = np.dot(A, A.transpose())
	if np.count_nonzero(cov - np.diag(np.diagonal(cov))) != 0:
		break 
u1 = random.rand(20)
while 1:
	u2 = random.rand(20)
	dist = np.linalg.norm(u1-u2)
	if dist < delta:
		break

fout = open('params.txt','w+')
fout.write('Mean vector for class A:\n')
str1 = ','.join(str(x) for x in u1)
fout.write('['+str1+']'+'\n')
fout.write('Mean vector for class B:\n')
str1 = ','.join(str(x) for x in u2)
fout.write('['+str1+']'+'\n')
fout.write('Co-variance Matrix:\n'+'[')
for y in cov:
	str1 = ','.join(str(x) for x in y)
	fout.write('['+str1+']'+'\n')
fout.write(']')

X1 = np.random.multivariate_normal(u1, cov, numDataPts)
X2 = np.random.multivariate_normal(u2, cov, numDataPts)
X = np.append(X1, X2, 0)
Y = [[1,0] for a  in range(numDataPts)]
Y.extend([[0,1] for a  in range(numDataPts)]) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
DS1_train = np.append(X_train, Y_train, 1)
DS1_test = np.append(X_test, Y_test, 1)
np.savetxt("../../Dataset/DS1-train.csv", DS1_train, delimiter=",")
np.savetxt("../../Dataset/DS1-test.csv", DS1_test, delimiter=",")
