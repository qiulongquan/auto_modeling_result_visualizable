#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from skopt import Optimizer
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import datetime
import numpy as np


def gpminimize_RandomForestRegressor(X, y):
    starttime = datetime.datetime.now()
    reg = RandomForestRegressor()
    # Define the hyperparameter configuration space
    space = [
        Integer(10, 100, name='n_estimators'),
        Integer(5, 50, name='max_depth'),
        Integer(1, 13, name='max_features'),
        Integer(2, 11, name='min_samples_split'),
        Integer(1, 11, name='min_samples_leaf'),
        Categorical(['mse', 'mae'], name='criterion')
    ]
    # Define the objective function

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)

        return -np.mean(
            cross_val_score(
                reg, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))

    res_gp_rf = gp_minimize(objective, space, n_calls=20, random_state=0)
    # number of iterations is set to 20, you can increase this number if time permits
    print("RandomForestRegressor MSE score:%.4f" % res_gp_rf.fun)
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_rf))
    print("最佳超参数值集合:", res_gp_rf.x)
    save_model_object(res_gp_rf.models, 'gp_minimize', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return res_gp_rf.fun, process_time_rf, res_gp_rf.x


def gpminimize_svr(X, y):
    starttime = datetime.datetime.now()
    reg = SVR(gamma='scale')
    space = [
        Real(1, 50, name='C'),
        Categorical(['poly', 'rbf', 'sigmoid'], name='kernel'),
        Real(0, 1, name='epsilon'),
    ]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(
            cross_val_score(
                reg, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))

    res_gp_svr = gp_minimize(objective, space, n_calls=20, random_state=0)
    print("SVR MSE score:%.4f" % res_gp_svr.fun)
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_svr))
    print("最佳超参数值集合:", res_gp_svr.x)
    save_model_object(res_gp_svr.models, 'gp_minimize', 'SVR', 'SVR')
    return res_gp_svr.fun, process_time_svr, res_gp_svr.x


def gpminimize_knn(X, y):
    starttime = datetime.datetime.now()
    reg = KNeighborsRegressor()
    space = [Integer(1, 20, name='n_neighbors')]

    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        return -np.mean(
            cross_val_score(
                reg, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))

    res_gp_knn = gp_minimize(objective, space, n_calls=10, random_state=0)
    print("KNN MSE score:%.4f" % res_gp_knn.fun)
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_knn))
    print("最佳超参数值集合:", res_gp_knn.x)
    save_model_object(res_gp_knn.models, 'gp_minimize', 'KNN', 'KNN')
    return res_gp_knn.fun, process_time_knn, res_gp_knn.x
