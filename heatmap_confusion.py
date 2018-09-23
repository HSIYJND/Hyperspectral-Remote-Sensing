#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:19:09 2018

@author: abhi
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rf_p = [[13174  ,  0 , 0 , 0 , 0 ,  21  ,   0  , 0 , 0] ,
     [0  ,   0  ,   0  ,   0  ,   0 , 1520 , 0 , 0 , 0],
    [ 0  ,   0  ,   0  ,   0  ,   0 ,  614  ,   0   ,  4  ,   0],
     [0  ,   0  ,   0  ,   0  ,   0  ,  16  ,   0  , 521  ,   0],
    [ 0  ,   0  ,   0  ,   0  ,   0  ,  33  ,   0 , 1284  ,   0],
     [4  ,   0  ,   0  ,   0  ,   0 , 1755  ,   0 ,   91  ,   0],
    [ 0  ,   0  ,   0  ,   0  ,   0 ,  969  ,   0 ,  489 ,    0],
     [0   ,  0  ,   0  ,   0 ,    0 ,  339  ,   0 , 8226 ,    0],
     [1  ,   0  ,   0  ,   0  ,   0  ,   2   ,  0 ,    0 ,  570]]

svm_p = [[13195  ,   0  ,   0  ,   0   ,  0  ,   0  ,   0   ,  0  ,   0],
     [0 , 1482  ,  38  ,   0   ,  0  ,   0  ,   0  ,   0 ,    0],
     [0  ,  36 ,  577 ,    0   ,  5   ,  0   ,  0 ,    0 ,    0],
    [ 0  ,   0  ,   0 ,  467 ,   41  ,   5  ,  21   ,  3 ,    0],
     [0  ,   1 ,    2  ,  46 , 1259  ,   0  ,   9  ,   0   ,  0],
     [0 ,    0   ,  0  ,  17 ,    2 , 1807  ,  19 ,    5  ,   0],
     [0 ,    0  ,   0  ,  25   ,  6  ,  68 , 1357 ,    2  ,   0],
     [0  ,   0  ,   0  ,  10 ,    2 , 14 ,    3 , 8537 ,    0],
     [0  ,   0  ,   0  ,   0  ,   0  ,   0 ,    0 ,    0 ,  572]]

knn_p = [[13194  ,   0  ,   0  ,   0  ,   0   ,  1   ,  0   ,  0   ,  0 ],
     [0 , 1459  ,  61 ,    0  ,   0  ,   0   ,  0   ,  0  ,   0],
     [0 ,   34 ,  581  ,   0  ,   3  ,   0  ,   0 ,    0  ,   0],
     [0  ,   0 ,    0  , 489  ,  21  ,   1  ,  22  ,   4   ,  0],
   [ 0  ,   0   ,  5  ,  34 , 1274  ,   0  ,   2  ,   2  ,   0],
    [ 1   ,  0  ,   0   ,  9  ,   0 , 1806 ,   25  ,   9  ,   0],
   [ 0  ,   0   ,  0  ,  35  ,   4  ,  39 , 1375  ,   5   ,  0],
    [ 0  ,   0   ,  0  ,  15  ,   7 ,   22  ,   3 , 8519  ,   0],
    [ 0  ,   0  ,   0  ,   0  ,   0 ,    0  ,   0  ,   0 ,  572]]


df = pd.DataFrame(rf_p)

df.columns = ['Water',	
	'Trees',	
	'Asphalt',	
	'Self-Blocking Bricks',	
	'Bitumen',	
	'Tiles',	
	'Shadows',	
	'Meadows',	
	'Bare Soil']

df.index = ['Water',	
	'Trees',	
	'Asphalt',	
	'Self-Blocking Bricks',	
	'Bitumen',	
	'Tiles',	
	'Shadows',	
	'Meadows',	
	'Bare Soil']


#result = df.pivot(index=a, columns=a, values='value')
#ax = sns.heatmap(df,cmap="YlGnBu" , annot=True, fmt="d")
ax = sns.heatmap(df,cmap="Spectral" )
ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
ax.set_xticklabels(ax.get_yticklabels(), rotation =60)
plt.title('Random Forest on Pavia Dataset')
plt.show()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matrix_pines = [[10 , 0 , 0 , 0 , 0 , 0 , 0  , 0  , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
	[0 , 227 ,  9 ,  0  , 1 ,  0 ,  0  , 0 ,  0 , 10 , 38 ,  1  , 0 ,  0 ,  0  , 0 ],
	[0 ,  8 , 134 , 5 ,  0  , 0  , 0 ,  0 , 0 , 0 , 13 ,  6 ,  0  , 0 ,  0 ,  0 ],
	[0 ,  5 ,  7 , 31 ,  2 ,  2 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ,  0  , 0 ,  0 , 0 ],
	[0 ,  0 ,  0 ,  2 , 94 ,  0 ,  0 , 0  , 0 ,  0 ,  0 ,  0 ,  0 ,  1 ,  0 ,  0 ],
	[0 ,  0 ,  0 ,  0 , 3 , 141 ,  0 ,  0 ,  0 ,  1 ,  0 ,  0 ,  0 ,  0 ,  1  , 0 ],
	[0 ,  0 ,  0 , 0 , 0  , 0 ,  5 ,  1 ,  0 ,  0 ,  0  , 0 ,  0 ,  0 ,  0 ,  0 ],
	[1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 95 , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  	0 ],
	[0 ,  0 ,  0 ,  0 ,  0 , 0 ,  0 ,  0 ,  4 ,  0  , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ],
	[0 , 14  , 1 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 138 , 41 ,  1 ,  0  , 0 ,  0  , 0 ],
	[0 , 28 , 27 , 1 ,  2 ,  2 ,  1 , 0 ,  0 , 25 , 394 , 11 ,  0 , 0 ,  0 ,  0 ],
	[0 ,  1  , 6 , 1 ,  2 ,  0 ,  0 , 0 , 0 ,  0 ,  8 , 101 , 0 ,  0 ,  0 ,  0 ],
	[0 ,  0 ,  0 , 0 ,  0  , 0 ,  0 , 0 ,  1  , 0 ,  0 , 0 , 40 , 0 ,  0 ,  0 ],
	[0 ,  0 ,  0 , 0 , 1 ,  4 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 236 , 12 , 0 ],
	[0 , 0 ,  0 , 0 , 3 ,  4 ,  0 ,  0 ,  2 ,  0 ,  1 ,  0 ,  1 , 11 , 56 , 0 ],
	[0 , 0 ,  0 , 0 , 0 , 0 ,  0 ,  0  , 0  , 0 ,  1 ,  0 , 0 ,  0 ,  0 , 18 ]]

knn_pines = [[  5 ,  0  , 0 ,  0  , 0 ,  0  , 0 ,  4  , 0 ,  0  , 1 ,  0 ,  0  , 0  , 0  , 0],
   [0 , 207 ,  8 ,  1  , 0 ,  0  , 0  , 0 ,  0 , 10 , 51 ,  9 ,  0  , 0 ,  0 ,  0],
   [0 , 18 , 97 ,  2 ,  0  , 0  , 0 ,  0 ,  0 ,  5 , 33 , 11  , 0 ,  0 ,  0 ,  0],
   [0 , 20 ,  3 , 16  , 0 ,  0 ,  0  , 0 ,  0  , 0 ,  8  , 1  , 0  , 0 ,  0  , 0],
   [0  , 1 ,  0 ,  1 , 85  , 4  , 1 ,  0 ,  0  , 1 ,  1 ,  1 ,  0 , 2 ,  0 ,  0],
   [0 ,  0 ,  0 ,  0 ,  0 , 143  , 0  , 0 ,  0  , 0 ,  0 ,  0 ,  0 ,  1  , 2  , 0],
   [1 ,  0  , 0  , 0  , 0 ,  0  , 5 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0  , 0  , 0  , 0],
   [0  , 0  , 0 ,  0  , 0  , 0 ,  0 , 96 ,  0 ,  0 ,  0  , 0  , 0  , 0  , 0 ,  0],
   [0  , 0  , 0  , 0  , 0 ,  1 ,  0  , 0 ,  2 ,  0 ,  0 ,  0  , 1 ,  0 ,  0 ,  0],
   [0  , 8 ,  6  , 0  , 2  , 0 ,  1 ,  0  , 0 , 149 , 26  , 3  , 0 ,  0 ,  0 ,  0],
   [0 , 33 , 23  , 4 ,  0  , 1 ,  2  , 1  , 0 , 31 , 393 ,  3  , 0  , 0 ,  0 ,  0],
   [0 , 20 ,  7  , 0  , 0 ,  0 ,  0 ,  0  , 0  , 8 , 19 , 65  , 0 ,  0  , 0  , 0],
   [0 ,  0 ,  0  , 0  , 0  , 2 ,  0  , 0  , 0  , 0  , 1  , 0 , 38 ,  0 ,  0  , 0],
   [0 ,  0  , 0 ,  0  , 5 ,  0  , 0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  2 , 241 , 5 , 0],
  [ 0 ,  0 ,  0 ,  0  , 6 , 19 ,  0  , 0 ,  1  , 2 ,  0  , 0 ,  5 , 17 , 28  , 0],
   [0 ,  0  , 0  , 0  , 0 ,  0  , 0  , 0 ,  0  , 0  , 1 ,  0  , 0 ,  0 ,  0 , 18]]

forest_pines =  [[ 0 ,  0  , 0  , 0  , 0 ,  9  , 0  , 0  , 0 ,  0 ,  1  , 0 ,  0  , 0  , 0  , 0],
   [0 , 102 ,  0  , 0  , 0 ,  4  , 0  , 0 ,  0 ,  0 , 180 ,  0 ,  0  , 0  , 0  , 0],
  [ 0 , 19 ,  0 ,  0  , 0 ,  0 ,  0  , 0  , 0 ,  0 , 147 ,  0 ,  0 ,  0 ,  0  , 0],
   [0 , 20 ,  0  , 0  , 0  , 6  , 0 ,  0 ,  0  , 0 , 22 ,  0  , 0  , 0 ,  0  , 0],
   [0 ,  1  , 0  , 0 ,  0 , 29  , 0 ,  0  , 0  , 0 ,  2 ,  0  , 0  , 65 ,  0  , 0],
   [0 ,  0 ,  0 ,  0  , 0 , 145 ,  0 ,  0 ,  0  , 0  , 1  , 0  , 0  , 0 ,  0  , 0],
   [0 ,  0  , 0  , 0  , 0 ,  6  , 0 ,  0  , 0 ,  0  , 0 ,  0  , 0  , 0 ,  0 ,  0],
   [0 ,  1  , 0  , 0  , 0 , 94 ,  0 ,  0  , 0  , 0  , 1  , 0 ,  0  , 0  , 0  , 0],
   [0  , 0 ,  0  , 0  , 0  , 4  , 0  , 0 ,  0  , 0 ,  0  , 0  , 0  , 0  , 0  , 0],
   [0 ,  0 ,  0 ,  0 ,  0 ,  7 , 0  , 0 ,  0 ,  0 , 188  , 0 ,  0 ,  0  , 0  , 0],
  [ 0 , 12  , 0 ,  0 ,  0 ,  8 ,  0 ,  0  , 0 ,  0 , 471  , 0 ,  0  , 0 ,  0 ,  0],
  [0  , 18 ,  0  , 0  , 0 ,  5 ,  0  , 0  , 0  , 0 , 96 ,  0 ,  0 ,  0  , 0  , 0],
  [0  , 0 ,  0  , 0  , 0 , 41 ,  0 ,  0 ,  0 ,  0  , 0 ,  0 ,  0  , 0 ,  0 ,  0],
   [0 ,  0  , 0  , 0  , 0 ,  8  , 0  , 0  , 0 ,  0 ,  0 ,  0 ,  0 , 245 ,  0 ,  0],
   [0  , 0 ,  0 ,  0 ,  0 , 54 ,  0  , 0  , 0  , 0  , 0 ,  0  , 0 , 24 ,  0 ,  0],
   [0  , 1 ,  0 ,  0  , 0  , 7  , 0 ,  0 ,  0 ,  0 , 11 ,  0  , 0  , 0  , 0 ,  0]]

df = pd.DataFrame(forest_pines)

df.columns = ['Alfalfa',	
'Corn-notill',
'Corn-mintill',
'Corn'	,
'Grass-pasture',	
'Grass-trees'	,
'Grass-pasture-mowed',	
'Hay-windrowed'	,
'Oats'	,
'Soybean-notill'	,
'Soybean-mintill',	
'Soybean-clean',	
'Wheat'	,
'Woods'	,
'Buildings-Grass-Trees-Drives',	
'Stone-Steel-Towers']

df.index = ['Alfalfa',	
'Corn-notill',
'Corn-mintill',
'Corn'	,
'Grass-pasture',	
'Grass-trees'	,
'Grass-pasture-mowed',	
'Hay-windrowed'	,
'Oats'	,
'Soybean-notill'	,
'Soybean-mintill',	
'Soybean-clean',	
'Wheat'	,
'Woods'	,
'Buildings-Grass-Trees-Drives',	
'Stone-Steel-Towers']


#ax = sns.heatmap(df,cmap="YlGnBu" , annot=True, fmt="d")
ax = sns.heatmap(df ,cmap="Spectral")
ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
ax.set_xticklabels(ax.get_yticklabels(), rotation =90)
plt.title('Random Forest on Indian Pines Data')
plt.show()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

svm_salinas = [[402  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0   , 0  ,  0   , 0 , 0   , 0], 
    [0  , 747  ,  0 ,   0  ,  0  ,  0  ,  0 , 0  ,  0 , 0 ,   0  ,  0 ,   0  ,  0 , 0  ,  0],
    [0  ,  0 , 395  ,  0  ,  0 ,   0  ,  0 ,   0  ,  0  ,  0   , 0  ,  0  ,  0   , 0 , 0  ,  0],
    [0  ,  0 ,   0 , 280 , 0  ,  0 , 0  ,  0  ,  0 ,   0 ,   0  ,  0 ,   0 ,   0,
     0 ,   0],
    [0  ,  0   , 0   , 1 , 534  ,  0  ,  0  ,  0  ,  0   , 0  ,  0  ,  0  ,  0   , 0 , 0  ,  0],
    [0  ,  0  ,  0  ,  0   , 0 , 792 , 0 , 0 ,   0  ,  0 ,   0 ,   0 ,   0 ,   0,
     0 ,   0],
    [0 ,   0  ,  0  ,  0  ,  0  ,  0 , 717  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 0 ,   0],
    [0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0 , 2255 , 0 ,   0 ,   0 ,   0 ,   0   , 0 , 0 ,   0],
    [0  ,  0  ,  0  ,  0 ,   0 ,   0  ,  0  ,  0 , 1240 ,   0 ,   0 ,   0  ,  0   , 0 , 0 ,   0],
   [ 0  ,  0  ,  0 ,   0 ,   0   , 0   , 0 ,   0  ,  0 , 655  ,  0  ,  0  ,  0   , 1 , 0 ,   0],
    [0  ,  0  ,  0  ,  0 ,   0  ,  0  ,  0 ,   0  ,  0  ,  0 , 213 ,   1  ,  0   , 0 , 0  ,  0],
   [ 0  ,  0   , 0  ,  0  ,  0 ,   0  ,  0  ,  0   , 0   , 0  ,  0 , 386 ,   0   , 0 , 0  ,  0],
   [ 0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0   , 0  ,  0 , 184 ,   0 , 0  ,  0],
   [ 0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0 ,   0  ,  0  ,  1  ,  0  ,  0  ,  0 , 213 , 0   , 0],
   [ 0   , 0  ,  0 ,   0 ,   0  ,  0  ,  0  ,  0  ,  0 ,   0   , 0  ,  0  ,  0   , 0 , 1454  ,  0],
   [ 0  ,  0  ,  0 ,   0  ,  0  ,  0  ,  0 ,   0 ,   0 ,   0  ,  0  ,  0 ,   0   , 0 , 0 , 362]]
    
rf_salinas=[[    0 ,  19  ,  0  ,  0  ,  0  ,  0  ,  0 , 383 ,   0  ,  0 ,   0  ,  0  ,  0   , 0 , 0 ,   0],
    [0 , 263 ,   0  ,  0   , 0 ,   0 ,   0 , 483  ,  0  ,  0  ,  0   , 0  ,  0   , 0 , 0  ,  0],
   [ 0  ,  0  ,  0  ,  0 , 0  ,  0  ,  0 , 396  ,  0 ,   0  ,  0  ,  0  ,  0 ,   0 , 0  ,  0],
   [ 0  ,  0  ,  0 ,   0  ,  0 ,   0 ,   0  , 162 ,   0  ,  0 ,   0 ,   0 ,   0   , 0 , 117 ,    0],
   [ 0 ,   0  ,  0  ,  0  ,  0  ,  2  ,  0 , 392  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 142  ,  0],
   [ 0  ,  0  ,  0  ,  0  ,  0 , 772  ,  0  , 19  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 2  ,  0 ],
   [ 0  ,  0  ,  0  ,  0  ,  0   , 0 , 336 , 143  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 236  ,  0],
   [ 0 ,   0  ,  0 ,   0   , 0  ,  0  ,  0 , 2222  ,  0 ,   0  ,  0 ,   0 ,  0  ,  0 , 
    34 ,   0],
    [0  ,  0   , 0  ,  0  ,  0  ,  0   , 0  ,  0 , 1240 ,   0 ,   0  ,  0  ,  0 ,   0 , 0  ,  0],
    [0  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 171 , 485   , 0 ,   0  ,  0 ,   0   , 0 , 0  ,  0],
   [ 0   , 0  ,  0  ,  0  ,  0 ,   0  ,  0 , 182 ,  32 ,   0  ,  0 ,   0  ,  0   , 0 , 0  ,  0],
   [ 0  ,  0 ,   0  ,  0  ,  0   , 0 ,   0 , 369 ,  17 ,   0 ,   0  ,  0 ,   0   , 0 , 0  ,  0],
   [ 0 ,   0  ,  0  ,  0  ,  0   , 0 ,   0 ,  85  , 96  ,  3  ,  0 ,   0  ,  0   , 0 , 0 ,   0],
   [ 0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0 , 176 ,  38  ,  0 ,   0   , 0   , 0   , 0 , 0 ,   0],
   [ 0  ,  0  ,  0  ,  0   , 0  ,  0  ,  0 ,  34 ,  0  ,  0 ,   0  ,  0  ,  0   , 0 ,  1420 ,   0],
   [ 0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  , 1 , 317 , 0  ,  0 ,   0  ,  0 ,   0 , 40  ,  4]]
   
knn_salinas =  [[402 ,   0  ,  0  ,  0  ,  0  ,  0  ,  0 ,   0  ,  0  ,  0  ,  0  ,  0  ,  0   , 0 , 0  ,  0],
    [0 , 745 ,   0  ,  0  ,  0  ,  0 ,   0 ,   1 ,   0  ,  0 ,   0  ,  0  ,  0  ,  0 , 0  ,  0],
    [0  ,  0 , 396 ,   0 ,   0  ,  0 ,   0 ,   0  ,  0  ,  0  ,  0  ,  0 ,   0   , 0 , 0  ,  0],
    [0 ,   0   , 0 , 272 ,   2  ,  0  ,  0  ,  1  ,  0  ,  0   , 0  ,  0  ,  0   , 0 , 4  ,  0],
    [0  ,  0  ,  0  ,  7 , 528  ,  1  ,  0  ,  0  ,  0  ,  0  ,  0 ,   0 ,   0   , 0 , 0  ,  0],
    [0  ,  0   , 0 ,   0  ,  0 , 791 ,   1 ,   0 ,   0 ,   0  ,  0  ,  0  ,  0   , 0 , 0  ,  0],
    [0  ,  0  ,  0  ,  0  ,  0  ,  0 , 716 ,   0  ,  0 ,   0  ,  0  ,  0  ,  0   , 0 , 0 ,   0],
    [0 ,   0  ,  0  ,  2  ,  0  ,  0 ,   0 , 2250 , 0 ,   0 ,   0 ,   0 ,   0 , 0 , 3  ,  0],
    [0  ,  0  ,  0  ,  0 ,   0  ,  0 ,   0  ,  0 , 1241  ,  0  ,  0  ,  0  ,  0   , 0 , 0  ,  0],
    [0  ,  0  ,  0  ,  0 ,   0  ,  0  ,  0 ,   0 , 0 , 654 ,   0 ,   0 ,   0  ,  2 , 0  ,  0],
    [0  ,  0 ,   0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0 , 211 ,   3 ,   0   , 0 , 0  ,  0],
    [0  ,  0   , 0  ,  0 ,   0  ,  0  ,  0 ,   0 ,   0  ,  0  ,  2 , 381  ,  4   , 0 , 0  ,   0],
    [0  , 0  ,  0 ,   0 ,   0  ,  0  ,  0  ,  0  ,  0 ,   0 ,   0  ,  1 , 179   , 3 , 0  ,  0],
    [0  ,  0 ,   0 ,   0 ,   0  ,  0  ,  0 ,   0 ,   0  ,  2 ,   0 ,   0  ,  5 , 207 , 0  ,  0],
    [0  ,  0 ,   0 ,   0  ,  0  ,  0  ,  0   , 0 ,   0 ,   0  ,  0  ,  0  ,  0   , 0 , 1454  ,  0],
    [0  ,  0  ,  0  ,  0  ,  0  ,  0  ,  0 ,   0  ,  0  ,  0  ,  0   , 0 ,   0    , 0 , 0 , 362]] 
    
df = pd.DataFrame(knn_salinas)

df.columns = ['Brocoli_green_weeds_1',	
	'Brocoli_green_weeds_2'	,
	'Fallow	',
	'Fallow_rough_plow'	,
'Fallow_smooth'	,
	'Stubble',	
'Celery',	
	'Grapes_untrained',	
	'Soil_vinyard_develop',	
	'Corn_senesced_green_weeds',	
	'Lettuce_romaine_4wk',	
	'Lettuce_romaine_5wk',	
	'Lettuce_romaine_6wk',	
	'Lettuce_romaine_7wk',
	'Vinyard_untrained'	,
	'Vinyard_vertical_trellis']
df.index = ['Brocoli_green_weeds_1',	
	'Brocoli_green_weeds_2'	,
	'Fallow	',
	'Fallow_rough_plow'	,
'Fallow_smooth'	,
	'Stubble',	
'Celery',	
	'Grapes_untrained',	
	'Soil_vinyard_develop',	
	'Corn_senesced_green_weeds',	
	'Lettuce_romaine_4wk',	
	'Lettuce_romaine_5wk',	
	'Lettuce_romaine_6wk',	
	'Lettuce_romaine_7wk',
	'Vinyard_untrained'	,
	'Vinyard_vertical_trellis']


#ax = sns.heatmap(df,cmap="YlGnBu" , annot=True, fmt="d")
ax = sns.heatmap(df ,cmap="Spectral")
ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
ax.set_xticklabels(ax.get_yticklabels(), rotation =90)
plt.title('kNN on Salinas Data')
plt.show()











pines = ['Alfalfa',	
'Corn-notill',
'Corn-mintill',
'Corn'	,
'Grass-pasture',	
'Grass-trees'	,
'Grass-pasture-mowed',	
'Hay-windrowed'	,
'Oats'	,
'Soybean-notill'	,
'Soybean-mintill',	
'Soybean-clean',	
'Wheat'	,
'Woods'	,
'Buildings-Grass-Trees-Drives',	
'Stone-Steel-Towers']

salinas = ['Brocoli_green_weeds_1',	
	'Brocoli_green_weeds_2'	,
	'Fallow	',
	'Fallow_rough_plow'	,
'Fallow_smooth'	,
	'Stubble',	
'Celery',	
	'Grapes_untrained',	
	'Soil_vinyard_develop',	
	'Corn_senesced_green_weeds',	
	'Lettuce_romaine_4wk',	
	'Lettuce_romaine_5wk',	
	'Lettuce_romaine_6wk',	
	'Lettuce_romaine_7wk',
	'Vinyard_untrained'	,
	'Vinyard_vertical_trellis']

pavia = ['Water',	
	'Trees',	
	'Asphalt',	
	'Self-Blocking Bricks',	
	'Bitumen',	
	'Tiles',	
	'Shadows',	
	'Meadows',	
	'Bare Soil']
