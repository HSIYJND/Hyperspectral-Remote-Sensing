#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:16:21 2018

@author: abhi
"""
#train the image on the train/test splitted subset
#apply the estimator on the whole data
#reshape the predicted values as per the gt

pred1= clf.predict(I_vect) 
error_image = pred1.reshape(145,145)

#1096x715 for Pavia
#512x217 for Salinas