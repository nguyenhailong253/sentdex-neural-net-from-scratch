### Calculating loss with categorical cross-entropy
## Before we do backpropagation, we need some sort of metric to measure how wrong is the model
## Let's use LOSS

import numpy as np
import math

# using logs
b = 5.2
print(np.log(b)) # natural log - log base e (euler number)
print(math.e**1.6486586255873816) # double check the value, expecting 5.2

# categorical cross-entropy example

softmax_output = [0.7, 0.1, 0.2] # made-up output that you might get - each output is a probability distribution, aka confidence (in %)
target_output = [1, 0, 0] # 1 hot vector
target_class = 0 # at index 0, it is "hot"

loss = -(math.log(softmax_output[0]) * target_output[0] + 
        math.log(softmax_output[1]) * target_output[1] +
        math.log(softmax_output[2]) * target_output[2])

print(loss)

loss = -math.log(softmax_output[0]) # since 1 is only "hot" at index 0, the other indices cancels out (by timing 0)
print(loss)
# softmax_output[0] in there, is considered 'confidence'. Confidence is within [0, 1]
# loss is within (-inf, 0]
# the higher the confidence, the lower the loss
# the lower the confidence, the higher the loss