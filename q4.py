# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:59:43 2019

@author: liujiaxuan
"""
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection  import cross_val_score
import numpy as np

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('mnist-original', data_home='./')
images = mnist.data
targets = mnist.target
X = mnist.data / 255.
Y = mnist.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

model= SVC(
            C=1.0, 
            kernel='rbf', 
            degree=3, 
            gamma='auto', 
            coef0=0.0, 
            shrinking=True, 
            probability=False, 
            tol=0.001, 
            cache_size=200, 
            class_weight=None, 
            verbose=False, 
            max_iter=-1, 
            decision_function_shape='ovr', 
            random_state=None)


"""
model = SVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,
                  C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                  class_weight=None, verbose=0, random_state=None, max_iter=1000)
"""
model.fit(X_train, Y_train)

train_accuracy = np.mean(cross_val_score(model, X_train, Y_train, cv=5))
test_accuracy = np.mean(cross_val_score(model, X_test, Y_test, cv=5))


print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))