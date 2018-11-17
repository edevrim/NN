#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:22:04 2018

@author: salihemredevrim
"""

#Neural Networks with Back Propagation 

import numpy as np 

import NN_BP

#X -> inputs
X = np.array(([1,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0],
              [0,0,0,1,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1]), dtype=int)



#output ys are equal to xs in this case 
y = X;

#Trials
#Neural_Network(X, y, n_input, n_output, n_hidden, learning_rate, lambda1, func, iter_num)
#Run 1
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.01, 10, 1, 100); 

#Run 2
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.01, 1, 1, 1000); 

#Run 3
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.1, 1, 10000); 

#Run 4
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.0, 1, 10000); 

#Run 5
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.5, 0.0, 1, 10000); 

#Run 6
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.5, 0.01, 1, 10000); 

#Run 7
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.01, 0.1, 1, 100000); 

#Run 8
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.001, 0.01, 1, 100000); 

#Run 9 
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.0, 1, 100000); 

#Run 10
NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.0001, 1, 1000000); 

#Run 11 
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.5, 0.0001, 0, 1000); 

#Run 12
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.001, 0, 0, 100000);

#Run 13
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.1, 0.0001, 0, 100000);

#Run 14
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.1, 0.001, 1, 100000);

#Run 15
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.001, 1, 1000000);

#Run 16
#NN_BP.Neural_Network(X, y, 8, 8, 3, 0.5, 0.001, 1, 1000000);