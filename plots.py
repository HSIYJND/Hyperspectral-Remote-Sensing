#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:24:57 2018

@author: abhi
"""

import matplotlib.pyplot as plt
import numpy as np

#AVIRIS-NG dataset
svml= [98.58,98.71,98.71,99.27]
svmp = [97.69,99.32,99.63,99.87]
svmr=[98.61,98.43,99.63,99.87]
knn= [94.27,96.46,96.46,97.2]
mlr= [97.09, 99.81,99.81,99.87]
rf=[95.76,96.2,97.62,98.29]

#Indian Pines dataset
svml= [72.84,81.16,85.87,84.62]
svmp = [68.61,76.46,82.34,87.13]
svmr=[75.13,84.07,88.52,89.14]
knn= [62.78,65.39,68.31,70.41]
mlr= [72.94,78.39,80.54,81.84]
rf=[73.62,82.37,85.87,88.32]

#salinas dataset
svml= [92.17,92.88,93.43,99.96]
#svmp = [90.73,81.28,92.60,84.64]
svmr=[92.01,90.09,89.54,94.85]
knn= [84.73,88.24,89.61,87.07]
mlr= [90.35,90.95,91.66,91.52]
rf=[91.32,94.68,94.73,95.1]

#pavia dataset
svml= [84.09,91.28,91.58,91.58]
svmp = [81.27,94.15,94.91,94.89]
svmr=[91.15,93.88,95.89,95.92]
knn= [85.75,89.02,89.27,89.96]
mlr= [87.96,88.73,88.78,88.96]
rf=[89.42,91.41,92.89,93.52]

plt.plot(svml)
plt.plot(svmp)
plt.plot(svmr)
plt.plot(knn)
plt.plot(mlr)
plt.plot(rf)
N=4
ind = np.arange(N)
plt.xticks(ind,('10%' ,'30%','60%','80%'))
plt.xticks(rotation=80)
plt.yticks(np.arange(90, 100.5, 5))
plt.ylabel('Overall Accuracy ')
plt.xlabel('Percentage of Training Samples')
plt.title('Classical Methods Comparison on AVIRIS-NG')
plt.legend(['SVM-linear', 'SVM-poly', 'SVM- rbf' ,'kNN' ,'MLR' ,'RF'], loc='top left')
plt.show()


#CNN graphs
N=4
ind = np.arange(N)
plt.bar(ind, height= [86.82,99.32,98.43,87.41])
plt.xticks(ind, ['1d-cnn',
'2d-cnn',
'3d-cnn',
'SVM-rbf'])
plt.yticks(np.arange(0, 100, 10))
plt.ylabel('Overall Accuracy ')
plt.xlabel('Various Architectures')
plt.title('DL models comparison on AVIRIS-NG data')
plt.show()







# These labels are only needed while showing class-wise accuracies
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