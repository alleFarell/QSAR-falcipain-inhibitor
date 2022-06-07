#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import copy
# from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time
# import multiprocessing as mp
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras import optimizers
# from keras.constraints import maxnorm
from numpy.random import seed
#from tensorflow import set_random_seed


# In[3]:


def r2_score(y, y_pred, y_train=None):
    sse = 0; sst = 0
    if y_train is None:        
        y_mean = np.mean(y)
    else:
        y_mean = np.mean(y_train)
    for i in range(len(y)):
        sse += (y[i] - y_pred[i]) ** 2
        sst += (y[i] - y_mean) ** 2
    r2_score = 1 - (sse / sst)
    return r2_score


# In[4]:


def q2_loo(model, X, y):
    loo = LeaveOneOut()
    y_pred = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        tmp = model.predict(X_test)
        #tmp = scaler_y.inverse_transform(tmp)
        y_pred.append(list(tmp)[0])
    r2 = r2_score(y_train, y_pred)
    return r2


# In[5]:


def leave_one_out(estimator, X, y, fit_time=False):
    n_data = X.shape[0]
    times = []; y_preds = []
    range_ = list(range(n_data))
    for i in range(n_data):
        train = np.setdiff1d(range_, i)
        X_tmp = X[train, :]
        y_tmp = y[train]
        mdl = estimator
        t0 = time.time()
        mdl.fit(X_tmp, y_tmp)
        time_ = time.time() - t0
        times.append(time_)
        y_pred = mdl.predict(X[i, :][np.newaxis, :])
        y_preds.append(np.asscalar(y_pred))
    fit_time = np.mean(times)
    if fit_time == True:
        return fit_time, y_preds
    else:
        return y_preds


# In[6]:


def mse_score(y, y_pred):
    if isinstance(y, np.ndarray) or isinstance(y, list):
        n_data = len(y)
        sum_err = 0
        for i in range(n_data):
            err = y[i] - y_pred[i]
            sum_err += (err ** 2)
        return sum_err / n_data
    else:        
        return (y - y_pred) ** 2


# In[7]:


def k_fold(estimator, X, y, n=5):
    kf = KFold(n_splits=n, random_state=42, shuffle=False)
    tmp_mse = []; tmp_r2 = []
    for train_index, test_index in kf.split(X):
        X_train_, X_test = X[train_index], X[test_index]
        y_train_, y_test = y[train_index], y[test_index]
        estimator.fit(X_train_, y_train_)
        y_pred = estimator.predict(X_test)
        r2_ = r2_score(y_test, y_pred)
        mse_ = mse_score(y_test, y_pred)
        if (np.isnan(mse_) or np.isinf(mse_)):
            tmp_mse.append(100)
        else:
            tmp_mse.append(mse_)
        if (np.isnan(r2_) or np.isinf(r2_)):
            tmp_r2.append(0)
        else:
            tmp_r2.append(r2_)
    sum_mse = np.average(tmp_mse)
    sum_r2 = np.average(tmp_r2)
    return sum_mse, sum_r2


# In[8]:


def loo_fit_predict(estimator, X_train, y_train, X_test, y_test, test_index):
    # standardize
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print(test_index)
    return y_pred


# In[9]:


def loo_cv_parallel(estimator, X, y):
    print("start")
    loo = LeaveOneOut()
    pool = mp.Pool(mp.cpu_count())
    y_pred = [pool.apply(loo_fit_predict, args=(estimator,X[train_index],y[train_index],X[test_index], 
                                                y[test_index], test_index)) for train_index, test_index in loo.split(X)]
    pool.close() 
    r2 = r2_score(y, y_pred)
    print(r2)
    return r2


# In[10]:


def k_fold_fit_predict(estimator, X_train, y_train, X_test, y_test):
    # standardize
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    r2 = r2_score(y_test, y_pred, y_train)
    mse = mse_score(y_test, y_pred)
#    for i in range(len(y_pred)):
#        print(y_test[i], y_pred[i])
    return mse, r2


# In[11]:


def k_fold_cv_parallel(estimator, X, y, n=5):
    kf = KFold(n_splits=n, random_state=42, shuffle=False)
    pool = mp.Pool(mp.cpu_count())
    tmp = [pool.apply(k_fold_fit_predict, args=(estimator,X[train_index],y[train_index],X[test_index], 
                                                y[test_index])) for train_index, test_index in kf.split(X)]
    pool.close() 
    tmp = [y for x in tmp for y in x]
    tmp_mse = []; tmp_r2 = []
    for i in range(len(tmp)):
        if i%2 == 0:
            tmp_mse.append(tmp[i])
        else:
            tmp_r2.append(tmp[i])
    sum_mse = np.average(tmp_mse)
    sum_r2 = np.average(tmp_r2)
    if (np.isnan(sum_mse) or np.isinf(sum_mse)):
        sum_mse = 100
    if (np.isnan(sum_r2) or np.isinf(sum_r2)):
        sum_r2 = 0
    return sum_mse, sum_r2


# In[12]:


def normalize(X):
    max_ = np.max(X, axis=0)
    min_ = np.min(X, axis=0)
    X_norm = (X - min_) / (max_ - min_)
    return max_, min_, X_norm


# In[13]:


def standardize(X):
    mean_ = np.mean(X, axis=0)
    std_ = np.std(X, axis=0)
    X_norm = (X - mean_) / std_
    return X_norm, mean_, std_


# In[14]:


def qsar_param(y, y_pred, d_r2m=True):
    results = []
    _, _, y = normalize(y)
    _, _, y_pred = normalize(y_pred)
    y_mean = np.mean(y); y_pred_mean = np.mean(y_pred)
    # calculate r2
    num = 0; den_1 = 0; den_2 = 0
    for i in range(len(y)):
        num += (y[i] - y_mean) * (y_pred[i] - y_pred_mean)
        den_1 += (y_pred[i] - y_pred_mean) ** 2
        den_2 += (y[i] - y_mean) ** 2
    r2 = num ** 2 / (den_1 * den_2)
    results = {"r2": r2}
    # calculate k and k_dash
    n_data = len(y)
    dot_ = 0; y_pred2 = 0; y2 = 0
    for i in range(n_data):
        dot_ += (y[i] * y_pred[i])
        y_pred2 += y_pred[i] ** 2
        y2 += y[i] ** 2
    k = np.sum(dot_) / np.sum(y_pred2)
    k_dash = np.sum(dot_) / np.sum(y2)
    results["k"] = k
    results["k_dash"] = k_dash
    # calculate r2_0 and r2_0_dash
    num = 0; num_dash = 0; den = 0; den_dash = 0
    for i in range(n_data):
        num += (y[i] - (k * y_pred[i])) ** 2
        num_dash += (y_pred[i] - (k_dash * y[i])) ** 2
        den += (y[i] - y_mean) ** 2
        den_dash += (y_pred[i] - y_pred_mean) ** 2
    r2_0 = 1 - (num / den)
    r2_0_dash = 1 - (num_dash / den_dash)
    #results.append(r2_0)
    #results.append(r2_0_dash)
    r2r0 = (r2 - r2_0)/r2
    r2r0_dash = (r2 - r2_0_dash)/r2
    results["r2r0"] = r2r0
    results["r2r0_dash"] = r2r0_dash
    r0r0_dash = np.abs(r2_0 - r2_0_dash)
    results["r0r0_dash"] = r0r0_dash
    # calculate rm2 and rm2_dash
    rm2 = r2 * (1 - np.sqrt(r2 - r2_0))
    rm2_dash = r2 * (1 - np.sqrt(r2 - r2_0_dash))
    results["rm2"] = rm2
    results["rm2_dash"] = rm2_dash
    #results.append(rm2)
    #results.append(rm2_dash)
    # calculate rm2_bar and d_rm2
    rm2_bar = (rm2 + rm2_dash) / 2
    d_rm2 = np.abs(rm2 - rm2_dash)
    results["rm2_bar"] = rm2_bar
    results["d_rm2"] = d_rm2
    return results


# In[15]:


def y_random(estimator, X, y, n=10):
    # non-random
    estimator.fit(X, y)
    y_pred = estimator.predict(X)    
    r2_nr = r2_score(y, y_pred)
    r_nr = np.sqrt(r2_nr)
    n_data = X.shape[0]
    # random
    r2_rand = []
    range_ = list(range(n_data))
    for i in range(n):
        new_range_ = copy.deepcopy(range_)
        np.random.shuffle(new_range_)
        y_new = []
        for i in new_range_:
            y_new.append(y[i])
        y_new = np.array(y_new)
        estimator.fit(X, y_new)
        y_pred = estimator.predict(X)
        r2_rand.append(r2_score(y_new, y_pred))
    r2_rand_avg = np.average(r2_rand)
    rp = r_nr * np.sqrt(r2_nr - r2_rand_avg)
    return rp


# In[16]:


def leverage(XtX, X):
    levs = []
    for i in range(X.shape[0]):
        x = X[i,:]
        lev = x.dot(XtX).dot(x.T)
        levs.append(lev)
    return levs


# In[17]:


def applicability_domain(X_train, X_test, y_train_act, 
                         y_train_pred, y_test_act, y_test_pred,fig_idx=0):
    # using wilson map
    X_train, _, _ = standardize(X_train)
    X_test, _, _ = standardize(X_test)
    n, p = X_train.shape
    # calculate standardized residuals
    err_train = []; res_train = []
    for i in range(len(y_train_pred)):
        err_train.append(y_train_act[i] - y_train_pred[i])
    rmse_train = np.sqrt(mse_score(y_train_act, y_train_pred))
    for i in range(len(y_train_pred)):
        tmp = err_train[i]/rmse_train
        res_train.append(tmp)
    err_test = []; res_test = []
    for i in range(len(y_test_pred)):
        err_test.append(y_test_act[i] - y_test_pred[i])
    rmse_test = np.sqrt(mse_score(y_test_act, y_test_pred))
    for i in range(len(y_test_pred)):
        tmp = err_test[i]/rmse_test
        res_test.append(tmp)    
    #res_test = [a/rmse_test for a in err_test]
    # calculate leverage
    XtX = X_train.T.dot(X_train)
    XtX = np.linalg.pinv(XtX)
    lev_train = leverage(XtX, X_train)
    lev_test = leverage(XtX, X_test)
    h_star = (3 * (p + 1)) / n
    print(h_star)
    #return (lev_train, res_train, lev_test, res_test, h_star)
#     print(lev_train)
#     print(res_train)
    #plotting
    import matplotlib.pyplot as plt
    plt.scatter(lev_train, res_train,marker='o', c='b', label='Train')
    plt.scatter(lev_test, res_test, marker='^', c='r', label='Test')
    plt.axhline(y=3, c='k', linewidth=0.8)
    plt.axhline(y=-3, c='k', linewidth=0.8)
    plt.axvline(x=h_star, c='k', linewidth=0.8, linestyle='dashed')
    #plt.xticks([0,0.1,0.2,0.3,0.4,0.5,h_star,0.6],[0,0.1,0.2,0.3,0.4,0.5,"h$^*$",0.6])
    plt.text(h_star+0.001, 0, "h$^*$")
    plt.xlim(0, h_star + 0.1)
    plt.ylim(-4, 4)
    plt.xlabel('leverage', fontname="Arial", fontsize=12)
    plt.ylabel('standardized residual', fontname="Arial", fontsize=12)
    plt.legend(prop={'family':"Arial", 'size':12}, loc='upper right')
    plt.savefig('./app_domain_{}.jpg'.format(fig_idx), format='jpg', dpi=1000, bbox_inches="tight")    
    plt.show()


# In[18]:


# Function to create model, required for KerasRegressor
def create_model_sa(input_dim=5):
    # create model
    seed(1)
    set_random_seed(2)
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='SGD')
    return model


# In[19]:


# class SimulatedAnnealing():
#     def __init__(self, t_init=100, t_fin=25, alpha=5, n_iter=100,
#                  rate=0.9, max_feat=1, seed=42):
#         self.t_init = t_init
#         self.t_fin = t_fin
#         self.alpha = alpha
#         self.n_iter = n_iter
#         self.seed = seed
#         self.rng = np.random.RandomState(seed)
#         self.max_feat = max_feat
#         self.rate = rate
#         self.idx = []
    
#     def stack_data(self, X, idx):
#         for i in range(len(idx)):
#             if i == 0:
#                 X_res = X[:, idx[i]][:, np.newaxis]
#             else:
#                 X_res = np.hstack((X_res, X[:, idx[i]][:, np.newaxis]))
#         return X_res
    
#     def check(self, idx, list_idx):
#         exist = False
#         for i in range(len(list_idx)):
#             tmp = list_idx[i]
#             check = all(elem in idx  for elem in tmp)
#             if check:
#                 exist = True
#         return exist
    
#     def compute_mse_r2(self, X, y):
#         """
#         if not self.idx: 
#             idx_ = self.rng.choice(np.arange(self.n_feat), self.max_feat,replace=False)
#             idx_ = np.sort(idx_).tolist()
#             self.idx.append(idx_)
#         else:
#             idx_ = self.rng.choice(np.arange(self.n_feat), self.max_feat,replace=False)
#             idx_ = np.sort(idx_).tolist()
#             flag = self.check(idx_, self.idx)
#             while flag:
#                 idx_ = self.rng.choice(np.arange(self.n_feat), self.max_feat,replace=False)
#                 idx_ = np.sort(idx_)
#                 flag = self.check(idx_, self.idx)
#             self.idx.append(idx_)
#         """ 
#         idx_ = self.rng.choice(np.arange(self.n_feat), self.max_feat,replace=False)
#         idx_ = np.sort(idx_).tolist()                    
#         X_tmp = self.stack_data(X, idx_)
#         """
#         np.random.seed(self.seed)
#         # build pipeline
#         estimators = []
#         estimators.append(('standardize', StandardScaler()))
#         estimators.append(('mlp', KerasRegressor(build_fn=create_model_sa, input_dim=self.input_dim, 
#                                                  epochs=100, batch_size=5, verbose=0)))
#         pipeline = Pipeline(estimators)
#         # cross validation
#         kfold = KFold(n_splits=10, random_state=self.seed)
#         cv_results = cross_validate(pipeline, X_tmp, y, cv=kfold, scoring=('r2', 'neg_mean_squared_error'), 
#                             return_train_score=True, n_jobs=-1)
#         mse = -np.mean(cv_results['test_neg_mean_squared_error'])
#         r2 = np.mean(cv_results['test_r2'])
#         """
#         estimator = KerasRegressor(build_fn=create_model_sa, input_dim=self.max_feat, epochs=100, batch_size=8, verbose=0)
#         mse, r2 = k_fold_cv_parallel(estimator, X_tmp, y, n=5)
#         return idx_, mse, r2
    
#     def select(self, X, y, label):
#         self.n_feat = X.shape[1]
#         # initialization
#         idx, mse, r2 = self.compute_mse_r2(X, y)
#         t = self.t_init
#         mse_list = [mse]
#         r2_list = [r2]
#         t_list = [t]
#         while t >= self.t_fin:
#             print("\n {} - temperature: {}".format(self.max_feat, t))
#             for _ in tqdm(range(self.n_iter)):
#                 # new solution
#                 idx_new, mse_new, r2_new = self.compute_mse_r2(X, y)
#                 if mse_new <= mse:
#                     mse = mse_new
#                     idx = idx_new
#                     r2 = r2_new
#                 else:             
#                     err_ = mse_new - mse
#                     k_ = -(self.t_init * np.log(0.8)) / err_
#                     proba = np.exp(-(k_ * err_) / t)
#                     rand_ = self.rng.rand()
#                     if rand_ < proba:
#                         mse = mse_new
#                         idx = idx_new
#                         r2 = r2_new
#             # update t
#             t *= self.rate
#             t_list.append(t)
#             mse_list.append(mse)
#             r2_list.append(r2)
#             print("desc: {}; mse: {}; r2: {}".format(label[idx], mse, r2))
#         return idx, [t_list, mse_list, r2_list]


# In[20]:


# Function to create model, required for KerasRegressor
def create_model(input_dim=5, hidden_layer=1, hidden_node=5, dropout_rate=0, optimizer='SGD', learn_rate=0.01, 
                 momentum=0.0, activation='relu'):
    # create model
    seed(1)
    set_random_seed(2)
    model = Sequential()
    for i in range(hidden_layer):
        if i == 0:
            model.add(Dense(hn, input_dim=input_dim, activation=activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, seed=1))
        else:
            model.add(Dense(hn, activation=activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate, seed=1))
    model.add(Dense(1, activation='linear'))
    # Compile model
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=learn_rate, momentum=momentum)
    elif optimizer == 'RMSprop':
        opt = optimizers.RMSprop(lr=learn_rate)
    elif optimizer == 'Adagrad':
        opt = optimizers.Adagrad(lr=learn_rate)
    elif optimizer == 'Adadelta':
        opt = optimizers.Adadelta(lr=learn_rate)        
    elif optimizer == 'Adam':
        opt = optimizers.Adam(lr=learn_rate)                
    elif optimizer == 'Adamax':
        opt = optimizers.Adamax(lr=learn_rate)                
    else:
        opt = optimizers.Nadam(lr=learn_rate)                
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


# In[21]:


def conv_list(y):
    res = []
    for i in range(len(y)):
        res.append(y[i])
    return res


# In[ ]:




