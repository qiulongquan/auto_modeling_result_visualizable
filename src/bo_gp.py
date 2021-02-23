#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from skopt import Optimizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from keras.wrappers.scikit_learn import KerasRegressor
from ANN_model import ANN
from basic_config import verbose
import datetime


def bo_RandomForestRegressor(X, y):
    # Define the hyperparameter configuration space
    rf_params = {
        'n_estimators': Integer(10, 100),
        "max_features": Integer(1, 13),
        'max_depth': Integer(5, 50),
        "min_samples_split": Integer(2, 11),
        "min_samples_leaf": Integer(1, 11),
        "criterion": ['mse', 'mae']
    }
    starttime = datetime.datetime.now()
    clf = RandomForestRegressor(random_state=0)
    Bayes_rf = BayesSearchCV(clf,
                             rf_params,
                             cv=3,
                             n_iter=20,
                             scoring='neg_mean_squared_error')
    # number of iterations is set to 20, you can increase this number if time permits
    Bayes_rf.fit(X, y)
    # bclf = Bayes_rf.best_estimator_
    print("RandomForestRegressor MSE score:" + str(-Bayes_rf.best_score_))
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_rf))
    print("最佳超参数值集合:", Bayes_rf.best_params_)
    save_model_object(Bayes_rf, 'BO-GP', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return str(-Bayes_rf.best_score_), process_time_rf, Bayes_rf.best_params_


def bo_svr(X, y):
    rf_params = {
        'C': Real(1, 50),
        "kernel": ['poly', 'rbf', 'sigmoid'],
        'epsilon': Real(0, 1)
    }
    starttime = datetime.datetime.now()
    clf = SVR(gamma='scale')
    Bayes_svr = BayesSearchCV(clf,
                              rf_params,
                              cv=3,
                              n_iter=20,
                              scoring='neg_mean_squared_error')
    Bayes_svr.fit(X, y)
    print("SVR MSE score:" + str(-Bayes_svr.best_score_))
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_svr))
    print("最佳超参数值集合:", Bayes_svr.best_params_)
    save_model_object(Bayes_svr, 'BO-GP', 'SVR', 'SVR')
    return str(
        -Bayes_svr.best_score_), process_time_svr, Bayes_svr.best_params_


def bo_knn(X, y):
    rf_params = {
        'n_neighbors': Integer(1, 20),
    }
    starttime = datetime.datetime.now()
    clf = KNeighborsRegressor()
    Bayes_knn = BayesSearchCV(clf,
                              rf_params,
                              cv=3,
                              n_iter=10,
                              scoring='neg_mean_squared_error')
    Bayes_knn.fit(X, y)
    print("KNN MSE score:" + str(-Bayes_knn.best_score_))
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_knn))
    print("最佳超参数值集合:", Bayes_knn.best_params_)
    save_model_object(Bayes_knn, 'BO-GP', 'KNN', 'KNN')
    return str(
        -Bayes_knn.best_score_), process_time_knn, Bayes_knn.best_params_


def bo_ANN(X, y):
    rf_params = {
        'activation': ['relu', 'tanh'],
        'loss': ['mse'],
        'batch_size': [32, 64, 128],
        'neurons': Integer(256, 1024),
        'epochs': [20, 30, 50, 60]
        # 'patience': Integer(3, 20)
    }
    starttime = datetime.datetime.now()
    clf = KerasRegressor(build_fn=ANN, verbose=verbose)
    Bayes_ann = BayesSearchCV(clf,
                              rf_params,
                              cv=3,
                              n_iter=10,
                              scoring='neg_mean_squared_error')
    Bayes_ann.fit(X, y)
    print("ANN MSE score:" + str(-Bayes_ann.best_score_))
    endtime = datetime.datetime.now()
    process_time_ann = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_ann))
    print("最佳超参数值集合:", Bayes_ann.best_params_)
    model_bo_ann = ANN(**Bayes_ann.best_params_)
    save_model_object(model_bo_ann, 'BO-GP', 'ANN', 'ANN')
    return str(
        -Bayes_ann.best_score_), process_time_ann, Bayes_ann.best_params_
