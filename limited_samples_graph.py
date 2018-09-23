"""
Created on Sun Sep 9 22:20:09 2018

@author: abhi
This file will mostly observe some trends in how the training sample size affects the classification.
We have already seen that smaller classes produce misclassification in the confusion matrix.

The limit after which Hughes effect is seen , also will be tested on the 16 class datasets of Indian Pines and Salinas,
and a smaller 9-class and more reliable dataset Pavia.
"""
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

pickle_in = open("/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Pickle_files/Forest_Pavia.p","rb")
rf = pickle.load(pickle_in)

pickle_in = open("/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Pickle_files/kNN_Pavia.p","rb")
kNN = pickle.load(pickle_in)

pickle_in = open("/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Pickle_files/SVM_Pavia.p","rb")
SVM = pickle.load(pickle_in)

pickle_in = open("/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Pickle_files/GMM_Pavia.p","rb")
GMM = pickle.load(pickle_in)
#print(mdl.predict_proba(rep))
#cross_val_score(mdl, I_vec , scoring='roc_auc', cv=10)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_vec = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/pavia.npy')
P_gt = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Paviagt.npy')

Pav_gt=np.ravel(P_gt)
Pav_vect=P_vec.transpose(2,0,1).reshape(102,-1).T

pav_sampleno=np.sort(Pav_gt)
pav_indices=np.argsort(Pav_gt)

#Array slicing to get each class of Pavia dataset
pgt=pav_sampleno[635488:]
P_new=pav_indices[635488:]

a=P_new[0:65971]
Class_1= Pav_vect[a]

b=P_new[65971:73569]
Class_2= Pav_vect[b]

c=P_new[73569:76659]
Class_3= Pav_vect[c]

d=P_new[76659:79344]
Class_4= Pav_vect[d]

e=P_new[79344:85928]
Class_5= Pav_vect[e]

f=P_new[85928:95176]
Class_6= Pav_vect[f]

g=P_new[95176:102463]
Class_7= Pav_vect[g]

h=P_new[102463:145288]
Class_8= Pav_vect[h]

i=P_new[145288:148151]
Class_9= Pav_vect[i]


#To split the ground truth according to the classes
gt_1=Pav_gt[a]
gt_2=Pav_gt[b]
gt_3=Pav_gt[c]
gt_4=Pav_gt[d]
gt_5=Pav_gt[e]
gt_6=Pav_gt[f]
gt_7=Pav_gt[g]
gt_8=Pav_gt[h]
gt_9=Pav_gt[i]


#Splitting training and testing data for each class for Indian Pines dataset
NUM=0.2

p_1, p_1_test, y_1, y_1_test = train_test_split(Class_1 ,gt_1, test_size=NUM)
p_2,p_2_test, y_2, y_2_test = train_test_split(Class_2 ,gt_2, test_size=NUM)
p_3, p_3_test, y_3, y_3_test = train_test_split(Class_3 ,gt_3, test_size=NUM)
p_4, p_4_test, y_4, y_4_test = train_test_split(Class_4 ,gt_4, test_size=NUM)
p_5, p_5_test, y_5, y_5_test = train_test_split(Class_5 ,gt_5, test_size=NUM)
p_6, p_6_test, y_6, y_6_test = train_test_split(Class_6 ,gt_6, test_size=NUM)
p_7, p_7_test, y_7, y_7_test = train_test_split(Class_7 ,gt_7, test_size=NUM)
p_8, p_8_test, y_8, y_8_test = train_test_split(Class_8 ,gt_8, test_size=NUM)
p_9, p_9_test, y_9, y_9_test = train_test_split(Class_9 ,gt_9, test_size=NUM)

X_train=np.concatenate((p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9))
y_train=np.concatenate((y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9))
X_test=np.concatenate((p_1_test,p_2_test,p_3_test,p_4_test,p_5_test,p_6_test,p_7_test,p_8_test,p_9_test))
y_test=np.concatenate((y_1_test,y_2_test,y_3_test,y_4_test,y_5_test,y_6_test,y_7_test,y_8_test,y_9_test))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mdl = SVM
import math
for i in (1,4):
    X_train =X_train[0:math.floor(len(X_train)/i)] 
    X_test =X_test[0:math.floor(len(X_test)/i)]
    y_train = X_train[0:math.floor(len(y_train)/i)]
    SVM.fit(X_train,y_train) 
    scr = SVM.score(X_train,y_train)
    scri =np.append(scr)
    pred=SVM.predict(X_test) 
    predd = np.append(pred)

plt.plot(scri)
plt.plot(predd)
plt.show()