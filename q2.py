# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:55:44 2019

@author: liujiaxuan
"""

from sklearn.naive_bayes import BernoulliNB
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


model = BernoulliNB()

train_accuracy = np.mean(cross_val_score(model, X_train, Y_train, cv=5))
test_accuracy = np.mean(cross_val_score(model, X_test, Y_test, cv=5))




print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))