# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:14:10 2020

@author: sandi
"""

# =============================================================================
# We will be building a neural network in pytorch from scratch
# =============================================================================

import torch

# Input tensor
X = torch.Tensor(
        [ [1.0, 0., 1., 0.],
          [1.0, 0., 1., 1.],
          [0., 1., 0, 1.] ],
        )

# output tensor
y = torch.Tensor(
        [ [1.0],
          [1.0],
          [0.0] ]
        )

batch_size = X.shape[0]
input_neurons = X.shape[1]
# No of hidden layer neurons
hidden_neurons = 3
# number of output neurons
output_neurons = 1

# epochs
num_epochs = 8000
# Learning rate 
learning_rate = 0.015


# weights and bias for input -> hidden
w_input_hidden = torch.randn((input_neurons, hidden_neurons))
b_input_hidden = torch.randn((hidden_neurons, 1))

# weights and bias for hidden -> output
w_hidden_output = torch.randn((hidden_neurons, output_neurons))
b_hidden_output = torch.randn((output_neurons, 1))

# sigmoid non linearity
def sigmoid(x):
    return 1 /( 1  + torch.exp(-x)) 

# derivative of sigmoid function
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Main loop 
for i in range(num_epochs):
    
    # Forward pass
    
    # Input -> Hidden
  
    hidden_layer_input = torch.mm(X, w_input_hidden) + b_input_hidden
    # Apply non linearity sigmoid
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Hidden -> Output
    
    output_layer_input = torch.mm(hidden_layer_output, w_hidden_output) + b_hidden_output
    # Added non linearity sigmoid
    output = sigmoid(output_layer_input)
    
    # mean squared is our loss function
    loss_function = ((y - output) ** 2) / 2
    
    # Print the loss every 1000 iterations
    if(i % 1000 == 0):
        print("Loss : %.10lf" % loss_function.mean())
    
    # Error function, derivative of loss ie d(Loss)/d(output)
    error = (output - y)
    
    # d(output) / d(output_layer_input)
    slope_output_layer = d_sigmoid(output)
    # d(hidden_layer_output) / d(hidden_layer_input)
    slope_hidden_layer = d_sigmoid(hidden_layer_output)
    
    # d(Loss) / d(output_layer_input) by chain rule
    d_output_layer_input = error * slope_output_layer
    
    # d(Loss) / d(hidden_layer_input) computed
    error_at_hidden_layer = torch.mm(d_output_layer_input, w_hidden_output.t())    
    d_hidden_layer = error_at_hidden_layer * slope_hidden_layer
    
    # Gradient Descent on hidden layer
    w_hidden_output -= torch.mm(hidden_layer_output.t(), d_output_layer_input) * learning_rate
    b_hidden_output -= d_output.sum() * learning_rate
    
    # Gradient Descent on output layer
    w_input_hidden -= torch.mm(X.t(), d_hidden_layer) * learning_rate
    b_input_hidden -= d_output.sum() * learning_rate
    
print('actual\n', y)
print('predictions\n', output)

# =============================================================================
#OUTPUT:
#actual
# tensor([[1.],
#        [1.],
#        [0.]])
#predictions
# tensor([[0.9895],
#        [0.9595],
#        [0.0304]])
# =============================================================================