"""
This file contains the one run that can be performed to get the results of all the pixel based
classifiers that can also be extended to take in the spatial feature component.
The dataset chosen to do all the analysis is the Pavia University dataset.

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
from sklearn.linear_model import LogisticRegression

img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/PaviaU.mat')
img=img['paviaU']
imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/PaviaU_gt.mat')
gt=imgt['paviaU_gt']
plt.imshow(gt,cmap='gnuplot2')


img=img
img=img.astype('float32')
indx=img.shape
im=img/np.max(img)

################################# Extract Classes ###########################
imr=im.reshape(indx[0]*indx[1],indx[2],order='F')
gtr=gt.reshape(indx[0]*indx[1],order='F')

''' ---------------Data Partition------------------------'''
indd=np.unique(gtr)
indd=indd+1
trts=dict()
for i in indd[:-1]:
    gtx=gtr.copy()
    gtx[gtr!=i]=0
    ind=np.nonzero(gtx)[0]
    trts[i]=train_test_split(ind, train_size=0.1, random_state=42)

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

    


######################SVM####################################


# grid search
nfolds =2

Cs = [0.02, 0.05,0.5,1]
gammas = [0.02, 0.05,0.2]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs = -1)
#grid_search=svm.SVC(kernel='rbf')
grid_search.fit(xtrain,ytrain)

grid_search.score(xtrain,ytrain) 
pred=grid_search.predict(xtest) 

print("accuracy",accuracy_score(ytest, pred))
print("kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))
print(grid_search.best_params_)

filename = 'SVM_Pavia.p'
pickle.dump(grid_search, open(filename, 'wb'))


#########################RF#################################################
iiRF = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True , max_depth=2, random_state=42) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

iRF = GridSearchCV(estimator=iiRF, param_grid=param_grid, cv= 5)

data=iRF.fit(xtrain, ytrain)
pred=iRF.predict(xtest) 
scrdone = iRF.score(xtrain,ytrain)

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))
print(iRF.best_params_)

pred1= iRF.predict(imr) 
error_image = pred1.reshape(610,340)

import pickle
filename = 'Forest_Pavia.p'
pickle.dump(iRF, open(filename, 'wb'))



##########################kNN#########################################
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=[7,8,9,10,11,12]
param_grid={'n_neighbors':n_neighbors}
neigh = GridSearchCV(KNeighborsClassifier(n_neighbors=n_neighbors), 
                           param_grid, cv=nfolds, n_jobs = -1)

neigh.fit(xtrain,ytrain) 
pred=neigh.predict(xtest) 

print("accuracy",accuracy_score(ytest, pred))
print("kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))
print(neigh.best_params_)

pred1= neigh.predict(imr) 
error_image = pred1.reshape(610,340)

filename = 'kNN_Pavia.p'
pickle.dump(neigh, open(filename, 'wb'))

'''
########################GMM##########################################
from sklearn.mixture import GMM
n_components=[7,8,9,10,11,12]
param_grid={'n_components':n_components}
classifier = GridSearchCV(GMM(n_components=9, covariance_type='full', n_iter=50), 
                           param_grid, n_jobs = -1)
                   
classifier.fit(xtrain)
pred=classifier.predict(xtest) 
pred1= classifier.predict(imr) 
error_image = pred1.reshape(610,340)
plt.imshow(error_image)  


print(accuracy_score(ytest, pred))
print(classification_report(ytest, pred))
print(cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))

filename = 'GMM_Pavia.p'
pickle.dump(classifier, open(filename, 'wb'))
'''


##################Multinomial LR #############################################

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(xtrain,ytrain)
pred=clf.predict(xtest)
clf.predict_proba(xtrain[:2, :]) 

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))

gtr_pred=gtr.copy()
gtr_pred[testid]=pred
error_image = gtr_pred.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')