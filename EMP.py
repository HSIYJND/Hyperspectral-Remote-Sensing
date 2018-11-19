
"""
Created on Thu Oct 18 01:50:19 2018

@author: abhi
"""

import skimage
import scipy
import matplotlib.pyplot as plt
from skimage.morphology import erosion,dilation,reconstruction, disk
from skimage.morphology import square,opening
import numpy as np

img=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/Indian_pines.mat')
img=img['indian_pines_corrected']
imgt=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/Indian_pines_gt.mat')
gt=imgt['indian_pines_gt']

################################################################################
img=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/PaviaU.mat')
img=img['paviaU']
imgt=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/PaviaU_gt.mat')
gt=imgt['paviaU_gt']

################################################################################
img=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/Salinas.mat')
img=img['salinas_corrected']
imgt=scipy.io.loadmat('/home/abhi/Desktop/Files_abhi/matdata/Salinas_gt.mat')
gt=imgt['salinas_gt']

img=img
img=img.astype('float32')
im=img/np.max(img)

indx=img.shape


selem=disk(6)
a=erosion(img,selem)
a=a.reshape(indx[0],indx[1],indx[2])
plt.imshow(a[:,:,10])
imr=a

b=dilation(imr,selem)
b=b.reshape(indx[0],indx[1],indx[2])
plt.imshow(b[:,:,100])

c=opening(imr,square(3))
c=c.reshape(indx[0],indx[1],indx[2])
plt.imshow(c[:,:,1])

y_mask = np.cos(imr)
y_seed = y_mask.min() * np.ones_like(imr)
y_seed[0] = 0.5
y_seed[-1] = 0
y_rec = reconstruction(y_seed, y_mask)


imr=imr.reshape(indx[0]*indx[1],indx[2],order='F')
gtr=gt.reshape(indx[0]*indx[1],order='F')

''' ---------------Data Partition------------------------'''
from sklearn.model_selection import train_test_split
indd=np.unique(gtr)
indd=indd+1
trts=dict()
for i in indd[:-1]:
    gtx=gtr.copy()
    gtx[gtr!=i]=0
    ind=np.nonzero(gtx)[0]
    trts[i]=train_test_split(ind, train_size=0.3, random_state=42)

xtrain=np.zeros((1,indx[2]))
xtest=np.zeros((1,indx[2]))
trainid=0
testid=0
ytrain=0
ytest=0
for j in indd[:-1]:
    xtrainn=imr[trts[j][0],:]
    xtestt=imr[trts[j][1],:]
    inx=xtrainn.shape
    iny=xtestt.shape
    ytrainn=np.zeros(inx[0])+j
    ytestt=np.zeros(iny[0])+j          
    xtrain=np.vstack((xtrain,xtrainn))
    xtest=np.vstack((xtest,xtestt))
    ytrain=np.append(ytrain,ytrainn)
    ytest=np.append(ytest,ytestt)
    trainidd=trts[j][0]
    testidd=trts[j][1]
    trainid=np.append(trainid,trainidd)
    testid=np.append(testid,testidd)
xtrain=xtrain[1::,:]
xtest=xtest[1::,:]
ytrain=ytrain[1::]
ytest=ytest[1::]
trainid=trainid[1::]
testid=testid[1::]

xtrainm = xtrain.mean(axis=0)
xtrainstd = xtrain.std(axis=0)

xtrain-=xtrainm
xtrain/=xtrainstd

xtest-=xtrainm
xtest/=xtrainstd



'''
nfolds =10
Cs = [0.01, 0.02, 0.05,1]
gammas = [0.01, 0.02, 0.05,1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds, n_jobs = -1)
'''


from sklearn import svm
from sklearn.svm import SVC
grid_search=svm.SVC(kernel='linear')
grid_search.fit(xtrain,ytrain)

grid_search.score(xtrain,ytrain) 
pred=grid_search.predict(xtest)


from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score 

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))
print(grid_search.best_params_)

