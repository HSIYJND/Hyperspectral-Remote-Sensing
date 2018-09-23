"""
Created on Wed Sep 19 17:20:21 2018

@author: abhi
"""

import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pandas as pd

MS = [ 0, 36.36, 16.67, 39.39, 30, 5.26, 0,
100, 27.27, 0, 92.21, 87.88, 74.63, 27.27, 60, 69.57 ]

HyS = [100, 97.37, 100, 100, 100, 100, 100,
100, 100, 100, 99.85, 100, 100, 100, 100, 100]
HyS2 = [86,68,62,51,84,93,50,95,50,73,73,54,93,91,64,79]

N=16
ind = np.arange(N)
plt.plot(MS)
plt.plot(HyS)
plt.xticks(ind,('Alfalfa',	
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
'Bldgs-Grass-Trees-Drives',	
'Stone-Steel-Towers'))
plt.xticks(rotation=80)
plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Class Accuracy (%)')

plt.title('SVM on Indian Pines ')
plt.legend(['MS', 'HyS'], loc='upper left')
plt.show()


N=16
ind = np.arange(N)
plt.plot(MS)
plt.plot(HyS)
plt.plot(HyS2)

plt.xticks(ind,('Alfalfa',	
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
'Bldgs-Grass-Trees-Drives',	
'Stone-Steel-Towers'))
plt.xticks(rotation=80)
plt.yticks(np.arange(0, 101, 10))
plt.ylabel('Class Accuracy (%)')
#plt.title('SVM versus Autoencoder')
plt.legend(['MS(SVM)','HyS(SVM)', 'Autoencoder'], loc='upper left')
plt.show()




'''

df = pd.DataFrame(MS)

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

#df.columns = [0,20,40,60,80,100]
df.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
ax=sns.lineplot(x=df.index, y=[0,36.36,16.67,39.39,30,5.26,0,
100,27.27,0,92.21,87.88,74.63,27.27,60,69.57],
             hue="region", style="event",
             data=df)
ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
ax.set_xticklabels(ax.get_yticklabels(), rotation =60)
'''

'''
# Overall Accuracy graph
plt.scatter([62,99.78])
plt.xlabel('')
plt.ylabel('')
plt.title('Overall Accuracy graph')
plt.show()

# Kappa Coefficient Accuracy graph
x=0.57
y=0.9974
plt.plot(x, y)
plt.xlabel('')
plt.ylabel('')
plt.title('Kappa Coefficient graph')
plt.show()
'''