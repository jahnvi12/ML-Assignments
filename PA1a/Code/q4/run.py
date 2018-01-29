#Data Imputation
import numpy as np
from sklearn.preprocessing import Imputer
dataset = np.genfromtxt("raw_data.csv", delimiter=",")
dataset = dataset[:,5:]
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
dataset = imp.fit_transform(dataset) 
np.savetxt("../../Dataset/CandC-dataset.csv", dataset, delimiter=",")