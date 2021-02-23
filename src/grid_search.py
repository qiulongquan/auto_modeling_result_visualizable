#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from skopt import Optimizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope
from sklearn.ensemble import GradientBoostingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from ANN_model import ANN
import datetime
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
from basic_config import verbose


def grid_RandomForestRegressor(X, y):
    # Define the hyperparameter configuration space
    rf_params = {
        'n_estimators': [10, 20, 30],
        'max_features': ['sqrt', 0.5],
        'max_depth': [15, 20, 30, 50],
        'min_samples_leaf': [1, 2, 4, 8],
        "bootstrap": [True, False],
        "criterion": ['mse', 'mae']
    }
    starttime = datetime.datetime.now()
    clf = RandomForestRegressor(random_state=0)
    grid_rf = GridSearchCV(clf,
                           rf_params,
                           cv=3,
                           scoring='neg_mean_squared_error')
    grid_rf.fit(X, y)
    print(grid_rf.best_params_)
    print("RandomForestRegressor MSE score:" + str(-grid_rf.best_score_))
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_rf))
    print("最佳超参数值集合:", grid_rf.best_params_)
    save_model_object(grid_rf, 'grid_search', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return str(-grid_rf.best_score_), process_time_rf, grid_rf.best_params_


def grid_svr(X, y):
    # Define the hyperparameter configuration space
    svr_params = {
        'C': [1, 10, 100],
        "kernel": ['poly', 'rbf', 'sigmoid'],
        "degree": np.arange(1, 10, 1),
        "epsilon": [0.01, 0.1, 1]
    }
    starttime = datetime.datetime.now()
    clf = SVR()
    grid_svr = GridSearchCV(clf,
                            svr_params,
                            cv=3,
                            scoring='neg_mean_squared_error')
    grid_svr.fit(X, y)
    print("SVR MSE score:" + str(-grid_svr.best_score_))
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_svr))
    print("最佳超参数值集合:", grid_svr.best_params_)
    save_model_object(grid_svr, 'grid_search', 'SVR', 'SVR')
    return str(-grid_svr.best_score_), process_time_svr, grid_svr.best_params_


def grid_knn(X, y):
    knn_params = {'n_neighbors': [2, 3, 5, 7, 10]}
    starttime = datetime.datetime.now()
    clf = KNeighborsRegressor()
    grid_knn = GridSearchCV(clf,
                            knn_params,
                            cv=3,
                            scoring='neg_mean_squared_error')
    grid_knn.fit(X, y)
    print("KNN MSE score:" + str(-grid_knn.best_score_))
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_knn))
    print("最佳超参数值集合:", grid_knn.best_params_)
    save_model_object(grid_knn, 'grid_search', 'KNN', 'KNN')
    return str(-grid_knn.best_score_), process_time_knn, grid_knn.best_params_


def grid_ANN(X, y):
    ann_params = {
        "neurons": [256, 512, 1028, 2048],
        "batch_size": [64, 128, 256],
        "epochs": [40, 60, 80],
        # "activation": ['sigmoid', 'relu', 'tanh'],
        # "patience": [2, 5],
        "loss": ['mse']
    }
    starttime = datetime.datetime.now()
    clf = KerasRegressor(build_fn=ANN, verbose=verbose)
    grid_ann = GridSearchCV(clf,
                            ann_params,
                            cv=3,
                            scoring='neg_mean_squared_error')
    grid_ann.fit(X, y)
    print("ANN MSE score:" + str(-grid_ann.best_score_))
    endtime = datetime.datetime.now()
    process_time_ann = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_ann))
    print("最佳超参数值集合:", grid_ann.best_params_)
    model_grid_ann = ANN(**grid_ann.best_params_)
    save_model_object(model_grid_ann, 'grid_search', 'ANN', 'ANN')
    return str(-grid_ann.best_score_), process_time_ann, grid_ann.best_params_
