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

#Run 10
NN_BP.Neural_Network(X, y, 8, 8, 3, 0.05, 0.0001, 1, 1000000); 
