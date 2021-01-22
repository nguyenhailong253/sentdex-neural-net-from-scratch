import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

# normalisation
norm_base = sum(exp_values) 
norm_values = [] # probability distribution

for value in exp_values:
    norm_values.append(value/norm_base)

# print(norm_values)
# print(sum(norm_values))

## ===================== using numpy

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

## ===================== using batches

layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]

exp_values = np.exp(layer_outputs)

print(np.sum(layer_outputs, axis=1, keepdims=True)) # axis = 1 is for row (y)
# keepdims so that it returns same dimension

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

## ======================= part 5 cont

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons)) 

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3) # spiral data is just (x,y), 2 features

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])