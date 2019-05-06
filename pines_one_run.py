"""
This file contains the one run that can be performed to get the results of all the pixel based
classifiers that can also be extended to take in the spatial feature component.
The dataset chosen to do all the analysis is the Indian Pines dataset.

"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_corrected.mat')
img=img['indian_pines_corrected']
imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_gt.mat')
gt=imgt['indian_pines_gt']
plt.imshow(gt,cmap='gnuplot2')


img=img.astype('float32')
indx=img.shape
im=img/np.max(img)

imr=im.reshape(indx[0]*indx[1],indx[2],order='F')
gtr=gt.reshape(indx[0]*indx[1],order='F')


indd=np.unique(gtr)
indd=indd+1
trts=dict()
for i in indd[:-1]:
    gtx=gtr.copy()
    gtx[gtr!=i]=0
    ind=np.nonzero(gtx)[0]
    trts[i]=train_test_split(ind, train_size=0.8, random_state=42)

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

#------------------- Number of train test samples------------------------
nclass=np.zeros((len(indd)-1,2))
for i in range(0,len(indd)-1):
    nclass[i,0]=len(trts[i+1][0])
    nclass[i,1]=len(trts[i+1][1])


###################################################SVM##########################

nfolds =2
Cs = [1,5,10,20,50,100,500,800,1000]
gammas = [0.0002,0.002, 0.02,0.2,0.1,1,2]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds, n_jobs = -1)
grid_search.fit(xtrain,ytrain)

grid_search.score(xtrain,ytrain) 
pred=grid_search.predict(xtest) 

print(classification_report(ytest, pred))
print(grid_search.best_params_)
print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
filename = 'SVM_linear.p'
pickle.dump(grid_search, open(filename, 'wb'))

#gtr_pred=a.copy()
gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')
plt.show()

#############

nfolds =2
Cs = [1,5,10,20,50,75,100,500]
gammas = [0.0001,0.0002,0.002, 0.02,0.2,0.1,1,2]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds, n_jobs = -1)
grid_search.fit(xtrain,ytrain)

grid_search.score(xtrain,ytrain) 
pred=grid_search.predict(xtest) 


print(classification_report(ytest, pred))

print(grid_search.best_params_)
print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
filename = 'SVM_poly.p'
pickle.dump(grid_search, open(filename, 'wb'))

#gtr_pred=grid_search.copy()
gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')
plt.show()
######

nfolds =2
Cs = [1,5,10,20,50,100,500,800,1000]
gammas = [0.0002,0.002, 0.02,0.2,0.1,1,2]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs = -1)
grid_search.fit(xtrain,ytrain)

grid_search.score(xtrain,ytrain) 
pred=grid_search.predict(xtest) 


print(classification_report(ytest, pred))

print(grid_search.best_params_)
print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
filename = 'SVM_rbf.p'
pickle.dump(grid_search, open(filename, 'wb'))

#gtr_pred=grid_search.copy()
gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')

plt.show()
#########################RF#################################################
iiRF = RandomForestClassifier(n_jobs=-1,oob_score = True , random_state=42) 

param_grid = { 
    'n_estimators': [10,50,100,200,400,600,800,1000,1500],
    'max_features': ['auto', 'sqrt', 'log2']
}

iRF = GridSearchCV(estimator=iiRF, param_grid=param_grid, cv= 10)

iRF.fit(xtrain, ytrain)
pred=iRF.predict(xtest) 
scrdone = iRF.score(xtrain,ytrain)


print(classification_report(ytest, pred))

print(iRF.best_params_)
print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
filename = 'Forest.p'
pickle.dump(iRF, open(filename, 'wb'))

#gtr_pred=iRF.copy()
gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')
plt.show()
##########################kNN#########################################
nfolds=2
n_neighbors=[2,3,4,5,6,7,8,9,10,11,12,13,14]
#n_neighbors=[10,11,12,13,14]
param_grid={'n_neighbors':n_neighbors}
neigh = GridSearchCV(KNeighborsClassifier(n_neighbors=n_neighbors), 
                           param_grid, cv=nfolds, n_jobs = -1)

neigh.fit(xtrain,ytrain) 
pred=neigh.predict(xtest) 


print(classification_report(ytest, pred))

print(neigh.best_params_)
print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
filename = 'kNN.p'
pickle.dump(neigh, open(filename, 'wb'))

#gtr_pred=neigh.copy()
gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')
plt.show()
'''
########################GMM##########################################

from sklearn.mixture import GMM
classifier = GMM(n_components=9, covariance_type='full', n_iter=50)
                   
classifier.fit(xtrain)
pred=classifier.predict(xtest) 
pred1= classifier.predict(imr) 

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))

filename = 'GMM_pines.p'
pickle.dump(classifier, open(filename, 'wb'))
'''

##################Multinomial LR #############################################

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(xtrain,ytrain)
pred=clf.predict(xtest)
#yprob=clf.predict_proba(xtrain)[:,1]
#print(yprob)

print(classification_report(ytest, pred))

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))
#gtr_pred=clf.copy()

filename = 'MLR.p'
pickle.dump(clf, open(filename, 'wb'))

gtr[testid]=pred
error_image = gtr.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')
plt.show()