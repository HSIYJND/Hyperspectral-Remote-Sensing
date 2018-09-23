#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 13:05:08 2018

@author: abhi
"""

import sklearn.preprocessing as sp
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt

num_com = 8

I_vec = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')
I_vect=I_vec.transpose((2,0,1))

I_gt = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Indian_pines_gt.npy')
igt=np.ravel(I_gt)

# Expand the array for scale
array_expand = I_vect[:,0,:]
for i_row in range(1, I_vect.shape[1]):
    tempmatirx = I_vect[:,i_row,:]
    array_expand = np.hstack((array_expand,tempmatirx))
        
# Data normalization
array_expand_scaled = sp.scale(array_expand.T)

ica = FastICA(n_components = num_com)
array_ica = ica.fit_transform(array_expand_scaled)

x = array_ica.reshape(145,145,8)
y0 = x[:,:,0]
y1 = x[:,:,1]
y2 = x[:,:,2]
y3 = x[:,:,3]
y4 = x[:,:,4]
y5 = x[:,:,5]
y6 = x[:,:,6]
y7 = x[:,:,7]
#y8 = x[:,:,8]

plt.imshow(y0)
