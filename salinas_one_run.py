"""
This file contains the one run that can be performed to get the results of all the pixel based
classifiers that can also be extended to take in the spatial feature component.

"""
import pickle
import numpy as np
from ggplot import *
from scipy.io import loadmat 
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import *
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Salinas.mat')
img=img['salinas']
imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Salinas_gt.mat')
gt=imgt['salinas_gt']
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

#------------------- Number of train test samples------------------------
nclass=np.zeros((len(indd)-1,2))
for i in range(0,len(indd)-1):
    nclass[i,0]=len(trts[i+1][0])
    nclass[i,1]=len(trts[i+1][1])
nfolds =5

###################################SVM###################

nfolds =2
Cs = [1,5,10,20,50,100,500,800,1000]
gammas = [0.0002,0.002, 0.02,0.2,0.1,1,2]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=nfolds, n_jobs = -1)
grid_search.fit(xtrain,ytrain)


pred=grid_search.predict(xtest) 

print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))


probs = model.predict_proba(xtest)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytest, pred)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

filename = 'SVM_poly_salinas.p'
pickle.dump(linear_I, open(filename, 'wb'))


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
error_image = pred1.reshape(512,217)

import pickle
filename = 'Forest_salinas.p'
pickle.dump(iRF, open(filename, 'wb'))



##########################kNN#########################################
n_neighbors=[5,6,7,8,9,10,11,12]
param_grid={'n_neighbors':n_neighbors}
neigh = GridSearchCV(KNeighborsClassifier(n_neighbors=n_neighbors), 
                           param_grid, cv=nfolds, n_jobs = -1)

neigh.fit(xtrain,ytrain) 
pred=neigh.predict(xtest) 


print("Overall Accuracy",accuracy_score(ytest, pred))
print("Kappa",cohen_kappa_score(ytest, pred))
print(classification_report(ytest, pred))
print(confusion_matrix(ytest, pred))
print(neigh.best_params_)

pred1= neigh.predict(imr) 
error_image = pred1.reshape(512,217)

filename = 'kNN_salinas.p'
pickle.dump(neigh, open(filename, 'wb'))

'''
########################GMM##########################################


classifier = GMM(n_components=9, covariance_type='full', n_iter=50)
                   
classifier.fit(xtrain)
pred=classifier.predict(xtest) 
pred1= classifier.predict(img) 
error_image = pred1.reshape(512,217)
plt.imshow(error_image)  


print(accuracy_score(ytest, pred))
print(classification_report(ytest, pred))
print(cohen_kappa_score(ytest, pred))
print(confusion_matrix(ytest, pred))

filename = 'GMM_salinas.p'
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
#print(clf.best_params_)

gtr_pred=gtr.copy()
gtr_pred[testid]=pred
error_image = gtr_pred.reshape(indx[0],indx[1],order='F')
plt.imshow(error_image,cmap='gnuplot2')