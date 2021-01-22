import numpy as np

inputs = [1, 2, 3, 2.5] # shape = (4,), type 1D array in numpy, vector in math

# 2D array, shape = (2, 3), this is a matrix (a list of vectors)
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

# outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight, in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     outputs.append(neuron_output)

# print(outputs)


output = np.dot(weights, inputs) + biases
# Explanation:
# np.dot(weights, inputs) = [np.dot(weights[0], inputs), 
#     np.dot(weights[1], inputs), 
#     np.dot(weights[2], inputs)]
print(output)

# output = weight * input + bias (this is like a linear graph)