#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 9 22:54:26 2018

@author: abhi

In this code file , all the estimators will be checked and how they fare in their uncertainity estimates from their multiclass/binary classifier
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


def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    #isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    #sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name)]:
                      #(isotonic, name + ' + Isotonic'),
                      #(sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #if hasattr(clf, "predict_proba"):
            #prob_pos = clf.predict_proba(X_test)[:, 1]
        #else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(rf, "Random Forest", 1)

# Plot calibration curve for Linear SVC
plot_calibration_curve(kNN, "kNN", 2)

plt.show()