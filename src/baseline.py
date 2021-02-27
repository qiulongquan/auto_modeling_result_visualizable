#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from keras.wrappers.scikit_learn import KerasRegressor
from src.ANN_model import ANN
import optuna
import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
import datetime
import time
import numpy as np
import pandas as pd


def bs_random_forest_regressor(X, y):
    starttime = datetime.datetime.now()
    base_rf = RandomForestRegressor()
    score = cross_val_score(base_rf,
                            X,
                            y,
                            cv=3,
                            scoring='neg_mean_squared_error')
    base_rf_score = -score.mean()
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print(" RandomForestRegressor MSE score {}".format(-score.mean()))
    print("程序执行时间（秒）:{}".format(process_time_rf))
    save_model_object(base_rf, 'baseline', 'randomforest', 'randomforest')
    return base_rf_score, process_time_rf


def bs_svr(X, y):
    starttime = datetime.datetime.now()
    base_svr = SVR()
    score = cross_val_score(base_svr,
                            X,
                            y,
                            cv=3,
                            scoring='neg_mean_squared_error')
    base_svr_score = -score.mean()
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("SVR MSE score {}".format(-score.mean()))
    print("程序执行时间（秒）:{}".format(process_time_svr))
    save_model_object(base_svr, 'baseline', 'svr', 'svr')
    return base_svr_score, process_time_svr


def bs_KNN(X, y):
    starttime = datetime.datetime.now()
    base_knn = KNeighborsRegressor()
    score = cross_val_score(base_knn,
                            X,
                            y,
                            cv=3,
                            scoring='neg_mean_squared_error')
    base_knn_score = -score.mean()
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("KNN MSE score {}".format(-score.mean()))
    print("程序执行时间（秒）:{}".format(process_time_knn))
    save_model_object(base_knn, 'baseline', 'knn', 'knn')
    return base_knn_score, process_time_knn


def bs_ANN(X, y):
    starttime = datetime.datetime.now()
    base_ann = KerasRegressor(build_fn=ANN, verbose=0)
    score = cross_val_score(base_ann,
                            X,
                            y,
                            cv=3,
                            scoring='neg_mean_squared_error')
    base_ann_score = -score.mean()
    endtime = datetime.datetime.now()
    process_time_ann = endtime - starttime
    print("ANN MSE score {}".format(str(-score.mean())))
    print("程序执行时间（秒）:{}".format(process_time_ann))
    save_model_object(base_ann, 'baseline', 'ann', 'ann')
    return base_ann_score, process_time_ann
