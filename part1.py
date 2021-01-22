inputs = [1.2, 5.1, 2.1] # unique inputs of curr layer, output of the previous layer
weights = [3.1, 2.1, 8.7]
bias = 3 # each neuron has a unique bias

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
