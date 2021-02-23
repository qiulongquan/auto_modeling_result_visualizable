#!/usr/bin/env python
# -*- coding:utf-8 -*-

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from random import randrange as sp_randrange
from keras.wrappers.scikit_learn import KerasRegressor
from ANN_model import ANN
from basic_config import verbose
import datetime
import scipy.stats as stats


def rs_RandomForestRegressor(X, y):
    # Define the hyperparameter configuration space
    rf_params = {
        'n_estimators': sp_randint(10, 100),
        "max_features": sp_randint(1, 13),
        'max_depth': sp_randint(5, 50),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "criterion": ['mse', 'mae']
    }
    # number of iterations is set to 20, you can increase this number if time permits
    n_iter_search = 20
    starttime = datetime.datetime.now()
    clf = RandomForestRegressor(random_state=0)
    Random_rf = RandomizedSearchCV(clf,
                                   param_distributions=rf_params,
                                   n_iter=n_iter_search,
                                   cv=3,
                                   scoring='neg_mean_squared_error')
    Random_rf.fit(X, y)
    print("RandomForestRegressor MSE score:" + str(-Random_rf.best_score_))
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_rf))
    print("最佳超参数值集合:", Random_rf.best_params_)
    save_model_object(Random_rf, 'random_search', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return str(-Random_rf.best_score_), process_time_rf, Random_rf.best_params_


def rs_svr(X, y):
    rf_params = {
        'C': stats.uniform(0, 50),
        "kernel": ['poly', 'rbf', 'sigmoid'],
        "epsilon": stats.uniform(0, 1)
    }
    n_iter_search = 20
    starttime = datetime.datetime.now()
    clf = SVR(gamma='scale')
    Random_svr = RandomizedSearchCV(clf,
                                    param_distributions=rf_params,
                                    n_iter=n_iter_search,
                                    cv=3,
                                    scoring='neg_mean_squared_error')
    Random_svr.fit(X, y)
    print("SVR MSE score:" + str(-Random_svr.best_score_))
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_svr))
    print("最佳超参数值集合:", Random_svr.best_params_)
    save_model_object(Random_svr, 'random_search', 'SVR', 'SVR')
    return str(
        -Random_svr.best_score_), process_time_svr, Random_svr.best_params_


def rs_knn(X, y):
    rf_params = {
        'n_neighbors': sp_randint(1, 20),
    }
    n_iter_search = 10
    starttime = datetime.datetime.now()
    clf = KNeighborsRegressor()
    Random_knn = RandomizedSearchCV(clf,
                                    param_distributions=rf_params,
                                    n_iter=n_iter_search,
                                    cv=3,
                                    scoring='neg_mean_squared_error')
    Random_knn.fit(X, y)
    print("KNN MSE score:" + str(-Random_knn.best_score_))
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_knn))
    print("最佳超参数值集合:", Random_knn.best_params_)
    save_model_object(Random_knn, 'random_search', 'KNN', 'KNN')
    return str(
        -Random_knn.best_score_), process_time_knn, Random_knn.best_params_


def rs_ANN(X, y):
    rf_params = {
        'optimizer': ['adam'],
        'activation': ['relu', 'tanh'],
        'loss': ['mse'],
        # 'batch_size': [16, 32, 64],
        'neurons': sp_randint(256, 1024),
        'epochs': [10, 20, 30, 50],
        # 'epochs':[20,50,100,200],
        # 'patience': sp_randint(3, 20)
    }
    n_iter_search = 10
    starttime = datetime.datetime.now()
    clf = KerasRegressor(build_fn=ANN, verbose=verbose)
    Random_ann = RandomizedSearchCV(clf,
                                    param_distributions=rf_params,
                                    n_iter=n_iter_search,
                                    cv=3,
                                    scoring='neg_mean_squared_error')
    Random_ann.fit(X, y)
    print("ANN MSE score:" + str(-Random_ann.best_score_))
    endtime = datetime.datetime.now()
    process_time_ann = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_ann))
    print("最佳超参数值集合:", Random_ann.best_params_)
    model_random_ann = ANN(**Random_ann.best_params_)
    save_model_object(model_random_ann, 'random_search', 'ANN', 'ANN')
    return str(
        -Random_ann.best_score_), process_time_ann, Random_ann.best_params_
