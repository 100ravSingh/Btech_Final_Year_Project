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


def mean_absolute_percentage_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# In[3]:


# cross validation
def cv(random_state, poly_degree, model, problem="compressive", print_folds=True):
    interaction_only = False
    if problem.lower() == "compressive":
        data_file ='../Dataset/compressive_strength.csv'
        data = pd.read_csv(data_file)
        interaction_only = True

    elif problem.lower() == "tensile":
        data_file = '../Dataset/tensile_strength.csv'
        data = pd.read_csv(data_file)
        
    elif problem.lower() == "test2":
        data_file = '../Dataset/data2set.csv'
        data = pd.read_csv(data_file)    
    else:
        print("The problem has to be compressive or tensile12 or tensile2")
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
            #print("[fold {0}] r: {1:.5f}, rmse(MPa): {2:.5f}, mae(MPa): {3:.5f}, mape(%): {4:.5f}".
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


# In[9]:


from sklearn.svm import SVR
regressor = SVR


# In[4]:


def barplot(x_data, y_data, error_data, x_label, y_label, title):
    _, ax = plt.subplots()
    x = np.arange(1, len(x_data) + 1)

    ax.bar(x, y_data, color='#539caf', align='center')
    ax.errorbar(x, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2, elinewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.show()


# In[5]:


def lineplot(x_data, y_data, label, dashed=True, marker='o', color='blue', markersize=8, linewidth=1.5):
    if dashed:
        plt.plot(x_data, y_data, 'r--', marker=marker, markerfacecolor=color, markersize=markersize, color=color,
                 linewidth=linewidth, label=label)
    else:
        plt.plot(x_data, y_data, marker=marker, markerfacecolor=color, markersize=markersize, color=color,
                 linewidth=linewidth, label=label)


# In[6]:


def run_model(regressor, params, random_state, poly_degree, problem):
    model = regressor(**params)
    cv(random_state=random_state, poly_degree=poly_degree, model=model, problem=problem)


# In[ ]:





# In[7]:


def run_svr(random_state, poly_degree, kernel, C, epsilon, gamma, problem):
    print("Running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma))

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=poly_degree)
    if kernel == "poly":
        poly_degree = 1

    cv(random_state, poly_degree, model, problem=problem)
    print("Finished running SVR for %s data with "
          "random_state=%s, poly_degree=%s, kernel=%s, C=%s, epsilon=%s, gamma=%s\n" %
          (problem, random_state, poly_degree, kernel, C, epsilon, gamma))


# In[10]:


def random_search():
    #random state
    random_state_opts= np.array([0, 1, 2, 42])
    
    #poly_degree
    poly_degree_opts = np.arange(1, 5, 1)

    # kernel
    kernel_opts = np.array(["rbf", "rbf", "poly", "linear"])  # 3

    # C
    C_opts = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]) # 15
    C_linear_opts = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]) # 10

    # epsilon
    epsilon_opts = np.array([.01, .02, .03, .04, .05, .06, .07, .08, .09, 0.1])  # 10

    # gamma
    gamma_opts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 9

    for i in range(10): #trail purpose we choose 10 , you can choose more than 10 for more values.
        kernel = str(np.random.choice(kernel_opts))

        C = np.random.choice(C_opts)
        if kernel == "linear":
            C = np.random.choice(C_linear_opts)

        epsilon = np.random.choice(epsilon_opts)

        gamma = np.random.choice(gamma_opts)
        random_state=np.random.choice(random_state_opts)
        poly_degree=np.random.choice(poly_degree_opts)

        run_svr(random_state, poly_degree, kernel, C, epsilon, gamma, problem="compressive")


if __name__ == "__main__":
    random_search()


# In[11]:


if __name__ == "__main__":
    from sklearn.svm import SVR
    regressor = SVR
    params = {"kernel": "rbf", "C": 1000, "epsilon": 0.04, "gamma": 0.5}
    run_model(regressor, params,random_state=0,poly_degree=1, problem="compressive")


# In[12]:


if __name__ == "__main__":
    
    from sklearn.svm import SVR
    regressor = SVR
    params = {"kernel": "rbf", "C": 20, "epsilon": 0.01, "gamma": 0.9}
    run_model(regressor, params,random_state=0,poly_degree=1, problem="tensile")


# In[13]:


if __name__ == "__main__":
    
    from sklearn.svm import SVR
    regressor = SVR
    params = {"kernel": "rbf", "C": 2000, "epsilon": 0.03, "gamma": 0.9}
    run_model(regressor, params,random_state=0,poly_degree=2, problem="test2")


# In[ ]:




