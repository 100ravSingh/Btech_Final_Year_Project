#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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


# In[2]:


#Compressive_Strength
df = pd.read_csv('../Dataset/compressive_strength.csv')
df.head()


# In[3]:


#Tensile_Strength
dp = pd.read_csv('../Dataset/tensile_strength.csv')
dp.head()


# In[4]:


#Tensile_strength_data2set
dy = pd.read_csv('../Dataset/data2set.csv')
dy.head()


# In[5]:


def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# In[ ]:





# In[6]:


# cross validation

def cv(random_state, poly_degree, model, problem, print_folds=True):
    interaction_only = False
    
    if problem.lower() == "compressive":
        data_file ='../Dataset/compressive_strength.csv'
        data = pd.read_csv(data_file)
        interaction_only = True

    elif problem.lower() == "tensile":
        data_file ='../Dataset/tensile_strength.csv'
        data = pd.read_csv(data_file)
        
    elif problem.lower() == "test2":
        data_file ='../Dataset/data2set.csv'
        data = pd.read_csv(data_file)

    else:
        print("The problem has to be compressive or tensile")
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
        X_test = X[test_index]
        y_test = y[test_index]

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

        # y_train_pred = model.predict(X_train)
        # y_train_pred = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1))

        # y_train = y_scaler.inverse_transform(y_train)

        # Error measurements
        r_lcc = r2_score(y_test, y_pred) ** 0.5
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        # print("RMSE on train: %s" % mean_squared_error(y_train, y_train_pred) ** 0.5)
        # print("RMSE on test: %s" % rmse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        scores.append([r_lcc, rmse, mae, mape])
        #if print_folds:
           # print("[fold {0}] r: {1:.5f}, rmse(MPa): {2:.5f}, mae(MPa): {3:.5f}, mape(%): {4:.5f}".
                  #format(fold_index, scores[fold_index][0], scores[fold_index][1], scores[fold_index][2], scores[fold_index][3]))
        fold_index += 1
    scores = np.array(scores)
    # barplot(["R2", "RMSE", "MAE", "MAPE"], scores.mean(0), scores.std(0), "Metrics", "Values",
    #         "Performance with different metrics")
    print('k-fold mean:              ', scores.mean(0))
    print('k-fold standard deviation:', scores.std(0))

    # Running time
    print('Running time: %.3fs ' % (time.time() - start_time))
    return scores.mean(0)[1].item()


# In[7]:


def run_model(regressor, params, random_state, poly_degree, problem):
    model = regressor(**params)
    cv(random_state=random_state, poly_degree=poly_degree, model=model, problem=problem)


# In[8]:


from sklearn.neural_network import MLPRegressor

# run cross validation for MLP
def run_mlp(random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha, problem):
    
    print("Running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, solver=%s, max_iter=%s, alpha=%s" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha))
    hd_layers = (hd_layer_1, hd_layer_2, )
    if hd_layer_2 < 0:
        hd_layers = (hd_layer_1, )

    model = MLPRegressor(warm_start=False, random_state=random_state,
                         hidden_layer_sizes=hd_layers, solver=solver, max_iter=max_iter, alpha=alpha)
    
    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running MLP for %s data with "
          "random_state=%s, poly_degree=%s, hd_layer_1=%s, hd_layer_2=%s, solver=%s, max_iter=%s, alpha=%s" %
          (problem, random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha))


# In[9]:


#def random_search():
    
    #problem
    #problem_opts = np.array(["compressive","tensile","test2"])
    
    #solver
    #solver_opts = np.array(["lbfgs","sgd"])
    
    #alpha 
    #alpha_opts = np.array([0,0.0001])
    
    # random_state
    #random_state_opts = np.array([0, 1, 2, 42]) 
    
    # poly_degree
    #poly_degree_opts = np.arange(1, 5, 1)  # 4

    # hd layer size 1
    #hd_layer_1_opts = np.array([100, 200, 300, 400, 500, 1000, 2000])  # 7

    # hd layer size 2
    #hd_layer_2_opts = np.array([-1, 100, 200, 300, 400, 500])  # 6

    # max iter
    #max_iter_opts = np.array([100, 200, 300, 400, 500, 1000])  # 6

    #for i in range(20):
        #hd_layer_1 = np.random.choice(hd_layer_1_opts)

        #hd_layer_2 = np.random.choice(hd_layer_2_opts)

        #max_iter = np.random.choice(max_iter_opts)
        
        #random_state = np.random.choice(random_state_opts)
        
        #poly_degree = np.random.choice(poly_degree_opts)
        
        #alpha = np.random.choice(alpha_opts)
        
        #solver = np.random.choice(solver_opts)
        
        #problem = np.random.choice(problem_opts)

        
        #run_mlp(random_state, poly_degree, hd_layer_1, hd_layer_2, solver, max_iter, alpha, problem)


#if __name__ == "__main__":
    #random_search()


# In[10]:


#compressive
if __name__ == "__main__":
    run_mlp(random_state=0, poly_degree=3, hd_layer_1=300, hd_layer_2=100, solver="lbfgs", max_iter=1000, 
            alpha=0, problem="compressive")


# In[11]:


#data2set
if __name__ == "__main__":
    run_mlp(random_state=0, poly_degree=3, hd_layer_1=100, hd_layer_2=200, solver="lbfgs", max_iter=1000, 
            alpha=0.0001, problem="test2")


# In[12]:


#tensile
if __name__ == "__main__":
    run_mlp(random_state=0, poly_degree=2, hd_layer_1=300, hd_layer_2=300, solver="lbfgs", max_iter=200, 
            alpha=0.0001, problem="tensile")


# In[ ]:




