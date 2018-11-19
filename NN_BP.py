#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:04:30 2018

@author: salihemredevrim
"""

#Neural Networks with Back Propagation 

import numpy as np 
import pandas as pd

#Activation function (sigmoid)
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Derivation of sigmoid function
#from the derivation in the Implementation note (Backprop Help)
def sig_prime(output):
    return output * (1 - output) 

#Activation function (tanh)
def tanh(z):
    return (np.exp(z) - np.exp(-1*z))/(np.exp(z) + np.exp(-1*z))

#Derivation of tanh function
def tanh_prime(output):
    return (1 - np.square(output))

#Forward 
def activation(X_norm_2, W_1, W_2, func): 
#If func option is set as 1 then sigmoid function is used, if 0 then tanh is used, more could be added     
#X_norm_2 normalized and bias variable added version of X as it initialized below
#W_1 and W_2 are weights in the first and second layers     

#Activation from inputs to the hidden layer
    z = np.dot(X_norm_2, W_1);
    if func == 1:
        output_0 = sigmoid(z);
    elif func == 0: 
        output_0 = tanh(z);

#Activation from the hidden layer to outputs
#At first, bias term is added to the second layer 
    n_row_hid = np.size(output_0, axis=0); 
    bias_2 = np.array([1.] * n_row_hid);
    output_01 = np.column_stack((bias_2, output_0)); 

    z_out = np.dot(output_01, W_2);
    if func == 1: 
        output = sigmoid(z_out);
    elif func == 0: 
        output = tanh(z_out);
    return output, output_01 #final output and staging output in the hidden layer
    
#Back propagation 
def back_pro(y_norm, output, W_1, W_2, alpha, lambda1, n_row, X_norm_2, output_01):   
#y_norm is the normalized y 
#outputs come from the activation
#alpha is given learning rate
#lambda1 is for weight decay    
#n_row is the number of training instances 
    
#Errors in the outputs
#comes from the derivative of (y - h(x))^2    
#from Backprop Help: This cost function is often used both for classification and for regression problems    
    error = -(y_norm - output); 

#Delta for hidden to output error
#comes from Step 2
    delta_2 = error * sig_prime(output); 

#Delta for input to hidden error
#comes from Step 3    
#how the error distributed in hidden layer
    delta_1 = np.dot(delta_2, W_2.T) * (output_01) * (1 - output_01); 
#Since the first term was for the bias term of the second layer, we can ignore this

#Partial derivatives
    grad_W2 = np.dot(output_01.T, delta_2); 
    grad_W1 = np.dot(X_norm_2.T, delta_1[:,1:]); #bias term is excluded

#Weight updates 
    W_2 = W_2 - (alpha * ((grad_W2 / n_row) + (lambda1 * W_2)));
    W_1 = W_1 - (alpha * ((grad_W1 / n_row) + (lambda1 * W_1)));    
    return W_1, W_2


#Initialization and iterations
def Neural_Network(X, y, n_input, n_output, n_hidden, learning_rate, lambda1, func, iter_num):

#Input values are min-max normalized 0 and 1 
#No need to normalize them for this dataset but let's do that for other datasets 
    ax = 0; 
    bx = 1; 

    X_max = np.amax(X, axis = 0); 
    X_min = np.amin(X, axis = 0); 
    X_norm = (bx - ax) * (X - X_min) / (X_max - X_min) + ax;
                
    
#For sigmoid and tanh activation functions outputs are rescaled between (0,1) and (-1,1) respectively 
    if func == 0: 
        ay = -1; 
        by = 1; 
    elif func == 1:
        ay = 0; 
        by = 1; 
   
    y_max = np.amax(y, axis = 0); 
    y_min = np.amin(y, axis = 0); 
    y_norm = (by - ay) * (y - y_min) / (y_max - y_min) + ay;

#Bias variable added to X_norm as 1 
    n_row = np.size(X_norm, axis=0); 
    bias = np.array([1.] * n_row);
    X_norm_2 = np.column_stack((bias, X_norm)); 

#Sizes of layers 
    input_size = n_input;
 
    output_size = n_output; 

    hidden_size = n_hidden; 

#learning rate
    alpha = learning_rate; 

#Network creation 
#Initialize randomly W_1 for weights from input to hidden layer (between 0-1)
#Then divided by 10 to make them close to 0 
# +1 for bias term
    W_1 = np.random.rand(input_size + 1, hidden_size) * 0.1; 
#Initialize randomly W_2 for weights from hidden to output layer (between 0-1)
    W_2 = np.random.rand(hidden_size + 1, output_size) * 0.1;

#Iterations
    i = 0; 
    while i < iter_num:
        i += 1; 
        [output, output_01] = activation(X_norm_2, W_1, W_2, func);
        [W_1, W_2] = back_pro(y_norm, output, W_1, W_2, alpha, lambda1, n_row, X_norm_2, output_01); 
        print("iteration num: \n" + str(i)); 
        print("Error: \n" + str(np.mean(np.square(y_norm - output)))); # mean sum of squared errors  
        #print("Output: \n" + str(np.around(output, 2)));
        #print("z_hidden_2: \n" + str(np.around(z_hidden_2, 2)));
        #print("W_1: \n" + str(np.around(W_1, 2)));
        #print("W_2: \n" + str(np.around(W_2, 2)));
        if np.mean(np.square(y_norm - output)) <= 0.01:
            print("Last iteration: \n" + str(i)); 
            break 
    
    print("Reality: \n" + str(y_norm)); 
    print("Output: \n" + str(np.around(output, 2))); 
    print("Final Error: \n" + str(np.mean(np.square(y_norm - output)))); 
    print("Weights 1: \n" + str(np.around(W_1, 2)));
    print("Weights 2: \n" + str(np.around(W_2, 2)));
    
    df1 = pd.DataFrame(output); 
    df2 = pd.DataFrame(W_1); 
    df3 = pd.DataFrame(W_2); 
    
    writer = pd.ExcelWriter('NN.xlsx', engine='xlsxwriter')
    df1.to_excel(writer, sheet_name='Output')
    df2.to_excel(writer, sheet_name='Weights 1')
    df3.to_excel(writer, sheet_name='Weights 2')
    writer.save()

    
    
    
    