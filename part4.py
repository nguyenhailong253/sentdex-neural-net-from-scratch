import numpy as np

# inputs = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

# # layer 1, each array is a neuron
# weights = [
#     [0.2, 0.8, -0.5, 1],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]


# biases = [2, 3, 0.5]

# # layer 2
# weights2 = [
#     [0.1, -0.14, 0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13]
# ]

# biases2 = [-1, 2, -0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases # transpose weights matrix

# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

# ======================== Refactor to a class =======================

X = [ # standard in ML to name this X
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # size of single sample input, and how many neuron you want in this layer
        # weight value is preferrably in the range (-1, 1)
        # the smaller the better, avoid exploding the initial input as it passes through layers
        # hence, the 0.1 to scale down values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # no need for transposing anymore, we already flip row & col
        self.biases = np.zeros((1, n_neurons)) # tuple argument because this 1st arg is supposed to be shape size

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

print(0.1 * np.random.randn(4,3))

layer1 = Layer_Dense(len(X[0]), 5)
layer2 = Layer_Dense(5, 2) # output of layer 1 is input of layer 2, so n_inputs have to match 5

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)