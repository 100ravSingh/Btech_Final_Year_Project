#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import xgboost
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import *
import time
from optparse import OptionParser
import matplotlib
from matplotlib import pyplot as plt
from tikzplotlib import save as tikz_save
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot




model = XGBRegressor()

# cross validation
def cv(random_state, poly_degree, model, tester,problem, print_folds=True):
    interaction_only = False
    MEAN = []
    if problem.lower() == "compressive":
        data_file ='../Dataset/compressive_strength.csv'
        data = pd.read_csv(data_file)
        interaction_only = True

    elif problem.lower() == "tensile":
        data_file = '../Dataset/tensile_strength.csv'
        data = pd.read_csv(data_file)

    elif problem.lower() == "test2":
        data_file ='../Dataset/data2set.csv'
        data = pd.read_csv(data_file)
    else:
        print("The problem has to be compressive or tensile or test2")
        return

    data = data.values
    n_data_cols = np.shape(data)[1]
    n_features = n_data_cols - 1

    # retrieve data for features
    X = np.array(data[:, :n_features])
    y = np.array(data[:, n_features:])
    # split into 10 folds with shuffle
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    start_time = time.time()
    scores = []
    fold_index = 0

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = tester
        #y_test = y[test_index]

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = y_scaler.fit_transform(y_train)

        if poly_degree >= 1:
            poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            # print ('Total number of features: ', X_train.size)

        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        MEAN.append(y_pred)
        
    return np.average(MEAN)
        



def run_xgb(random_state, poly_degree, n_estimators, max_depth, learning_rate,objective, problem,tester):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         objective=objective, random_state=random_state)
    outpts = cv(random_state, poly_degree, model,tester, problem=problem)
    return outpts
    


def calci(testing,natures):
    if natures == "compressive":
        outpts2 = run_xgb(random_state=0, poly_degree=1, n_estimators=1400, max_depth=6,
        learning_rate=0.15, objective="reg:logistic", problem="compressive",tester=testing)
    
    if natures == "tensile":
        outpts2 = run_xgb(random_state=0, poly_degree=2, n_estimators=700, max_depth=6,
        learning_rate=0.09, objective="reg:logistic", problem="tensile",tester=testing)
        
    return outpts2


