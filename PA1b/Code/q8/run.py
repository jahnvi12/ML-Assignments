# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from sklearn.metrics import classification_report
np.random.seed(seed=1)
from sklearn import datasets, model_selection, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
from sklearn.preprocessing import normalize
#Set this flag to True for part1
part1 = False
Gamma = 0.001

dataset = np.genfromtxt("../../Data_LR(DS2)/data_students/Train_features8", skip_header = 2, delimiter=" ")
np.random.shuffle(dataset)
T_train = dataset[:,-4:]
dataset = np.delete(dataset,96,1)
dataset = np.delete(dataset,96,1)
dataset = np.delete(dataset,96,1)
X_train = np.delete(dataset,96,1)
dataset = np.genfromtxt("../../Data_LR(DS2)/data_students/Test_features8", skip_header = 2, delimiter=" ")
np.random.shuffle(dataset)
T_test = dataset[:,-4:]
dataset = np.delete(dataset,96,1)
dataset = np.delete(dataset,96,1)
dataset = np.delete(dataset,96,1)
X_test = np.delete(dataset,96,1)
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

X_validation, X_test, T_validation, T_test = model_selection.train_test_split(
    X_test, T_test, test_size=0.5)

def logistic(z): 
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y):
    return np.multiply(y, (1 - y))
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

class Layer(object):
    
    def get_params_iter(self):
        return []
    
    def get_params_grad(self, X, output_grad):
        return []
    
    def get_output(self, X):
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        pass

class LinearLayer(Layer):
   
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
        
    def get_params_iter(self):
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))
    def get_weight(self):
        return self.W
    def get_bias(self):
        return self.b

    def get_output(self, X):
        return X.dot(self.W) + self.b
        
    def get_params_grad(self, X, output_grad):
        if part1:
            JW = X.T.dot(output_grad)
            Jb = np.sum(output_grad, axis=0)
        else:
            JW = X.T.dot(output_grad) + (2*Gamma)*self.W
            Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        return output_grad.dot(self.W.T)


class LogisticLayer(Layer):
    
    def get_output(self, X):
        return logistic(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(logistic_deriv(Y), output_grad)


class SoftmaxOutputLayer(Layer):
    
    def get_output(self, X):
        return softmax(X)
    
    def get_input_grad(self, Y, T):
        if part1:
            return (Y - T) / Y.shape[0]
        else:
            ans = np.zeros(shape=Y.shape)
            for i in range(len(T)):
                for k in range(len(T[i])):
                    for j in range(len(T[i])):
                        delt = 0
                        if j == k:
                            delt = 1 
                        ans[i][k] += 2*T[i][j]*(delt-Y[i][k])*(Y[i][j] - T[i][j])
            return ans / Y.shape[0]
    
    def get_cost(self, Y, T):
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]

# MODEL
hidden_neurons_1 = 50  
layers = [] 
layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1))
layers.append(LogisticLayer())
layers.append(LinearLayer(hidden_neurons_1, T_train.shape[1]))
layers.append(SoftmaxOutputLayer())
iteration = 0

def forward_step(input_samples, layers):
    activations = [input_samples]
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  
        activations.append(Y)  
        X = activations[-1]  
    if iteration%1000 is 0:
        pass#print activations[-1]
    return activations  

def backward_step(activations, targets, layers):
    param_grads = collections.deque() 
    output_grad = None  
    for layer in reversed(layers):   
        Y = activations.pop()  
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else: 
            input_grad = layer.get_input_grad(Y, output_grad)
        X = activations[-1]
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        output_grad = input_grad
    return list(param_grads)

batch_size = 25  
nb_of_batches = X_train.shape[0] / batch_size 

XT_batches = zip(
    np.array_split(X_train, nb_of_batches, axis=0), 
    np.array_split(T_train, nb_of_batches, axis=0))  

def update_params(layers, param_grads, learning_rate):
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in itertools.izip(layer.get_params_iter(), layer_backprop_grads):
            param -= learning_rate * grad  
       
training_costs = []
validation_costs = []

max_nb_of_iterations = 700 
learning_rate = 0.05

for iteration in range(max_nb_of_iterations):
    for X, T in XT_batches:  
        activations = forward_step(X, layers)  
        param_grads = backward_step(activations, T, layers)  
        update_params(layers, param_grads, learning_rate)
    activations = forward_step(X_train, layers)
    train_cost = layers[-1].get_cost(activations[-1], T_train)
    training_costs.append(train_cost)
    activations = forward_step(X_validation, layers)
    validation_cost = layers[-1].get_cost(activations[-1], T_validation)
    validation_costs.append(validation_cost)
    if len(validation_costs) > 3:
        if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
            pass#break
    
nb_of_iterations = iteration + 1
y_true = np.argmax(T_test, axis=1)
activations = forward_step(X_test, layers)
y_pred = np.argmax(activations[-1], axis=1) 
np.savetxt('weights1.csv', layers[0].get_weight(), delimiter=',')
np.savetxt('weights2.csv', layers[2].get_weight(), delimiter=',')
test_accuracy = metrics.accuracy_score(y_true, y_pred) 
print('The accuracy on the test set is {:.2f}'.format(test_accuracy))
print "Classification Report: \n"
print(classification_report(y_true, y_pred))
