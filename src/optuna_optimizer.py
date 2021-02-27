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
from src.ANN_model import ANN
from src.basic_config import verbose
import datetime
import xgboost as xgb
import lightgbm as lgb
import optuna
import numpy as np

# NGBoost需要的包
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
import warnings


def optuna_RandomForestRegressor(X, y):
    # Define the objective function
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 1, 100)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        clf = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=0)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    # 因为我们要获得最好的MSE，所以方向是min。direction="minimize"
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective, n_trials=5)
    optuna_rf_mse_score = study_rf.best_value
    optuna_rf_time = (study_rf.best_trial.datetime_complete -
                      study_rf.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_rf_time, 60)
    h, m = divmod(m, 60)
    optuna_rf_time = "%d:%02d:%09f" % (h, m, s)
    print("RandomForestRegressor MSE score:%.4f" % optuna_rf_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_rf_time))
    print("最佳超参数值集合:", study_rf.best_params)
    save_model_object(study_rf, 'Optuna', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return optuna_rf_mse_score, optuna_rf_time, study_rf.best_params


def optuna_GradientBoostingRegressor(X, y):
    def objective(trial):
        #     设定了4个搜索范围subsample，n_estimators，max_depth，lr
        subsample = trial.suggest_discrete_uniform("subsample", 0.1, 1.0, 0.1)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
        clf = GradientBoostingRegressor(n_estimators=n_estimators,
                                        subsample=subsample,
                                        learning_rate=lr,
                                        max_depth=max_depth,
                                        random_state=0)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    study_name_gbr = 'optuna-gbr'  # Unique identifier of the study.
    study_gbr = optuna.create_study(direction="minimize",
                                    study_name=study_name_gbr)
    # 可以加载sqlite3的db数据库里面的信息
    # study = optuna.create_study(study_name='example-study', storage='sqlite:///example.db', load_if_exists=True)
    # 加载后直接优化模型
    study_gbr.optimize(objective, n_trials=5)
    optuna_gbr_mse_score = study_gbr.best_value
    optuna_gbr_time = (study_gbr.best_trial.datetime_complete -
                       study_gbr.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_gbr_time, 60)
    h, m = divmod(m, 60)
    optuna_gbr_time = "%d:%02d:%09f" % (h, m, s)
    print("GradientBoostingRegressor MSE score:%.4f" % optuna_gbr_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_gbr_time))
    print("最佳超参数值集合:", study_gbr.best_params)
    save_model_object(study_gbr, 'Optuna', 'GradientBoostingRegressor',
                      'GradientBoostingRegressor')
    return optuna_gbr_mse_score, optuna_gbr_time, study_gbr.best_params


def optuna_svr(X, y):
    def objective(trial):
        params = {
            "kernel":
            trial.suggest_categorical("kernel", ["linear", "poly", "rbf"]),
            "C":
            trial.suggest_loguniform("C", 1e-5, 1e2),
            # 'degree': trial.suggest_int("degree", 1,10,step=1),
            # 'epsilon': trial.suggest_float("epsilon", 0.1,0.5,step=0.1),
        }
        clf = SVR(**params, gamma="scale")
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    study_svr = optuna.create_study(direction="minimize")
    study_svr.optimize(objective, n_trials=5)
    optuna_svr_mse_score = study_svr.best_value
    optuna_svr_time = (study_svr.best_trial.datetime_complete -
                       study_svr.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_svr_time, 60)
    h, m = divmod(m, 60)
    optuna_svr_time = "%d:%02d:%09f" % (h, m, s)
    print("SVR MSE score:%.4f" % optuna_svr_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_svr_time))
    print("最佳超参数值集合:", study_svr.best_params)
    save_model_object(study_svr, 'Optuna', 'SVR', 'SVR')
    return optuna_svr_mse_score, optuna_svr_time, study_svr.best_params


def optuna_knn(X, y):
    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int("n_neighbors", 1, 20, step=1),
        }
        clf = KNeighborsRegressor(**params)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    study_knn = optuna.create_study(direction="minimize")
    study_knn.optimize(objective, n_trials=5)
    optuna_knn_mse_score = study_knn.best_value
    optuna_knn_time = (study_knn.best_trial.datetime_complete -
                       study_knn.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_knn_time, 60)
    h, m = divmod(m, 60)
    optuna_knn_time = "%d:%02d:%09f" % (h, m, s)
    print("KNN MSE score:%.4f" % optuna_knn_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_knn_time))
    print("最佳超参数值集合:", study_knn.best_params)
    save_model_object(study_knn, 'Optuna', 'KNN', 'KNN')
    return optuna_knn_mse_score, optuna_knn_time, study_knn.best_params


def optuna_ANN(X, y):
    # 官网optuna都是使用sklearn里面定义好的模型，自定义模型要想使用optuna比较复杂。
    # 一些参数使用默认就可以，不需要调整，默认值基本都是mes分数最低的

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5,
                                                      1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
            "activation": trial.suggest_categorical("activation",
                                                    ['relu', 'tanh']),
            'neurons': trial.suggest_int("neurons", 512, 2048, step=128),
            'epochs': trial.suggest_int("epochs", 40, 100, step=10),
        }

        clf = KerasRegressor(build_fn=ANN, **params, verbose=verbose)
        score = cross_val_score(clf,
                                X,
                                y,
                                cv=3,
                                scoring='neg_mean_squared_error')
        obtuna_ann_score = -score.mean()
        # 官网optuna都是使用sklearn里面定义好的模型，自定义模型要想使用optuna比较复杂。
        return obtuna_ann_score

    study_name_ann = 'optuna-ann'  # Unique identifier of the study.
    study_ann = optuna.create_study(direction="minimize",
                                    study_name=study_name_ann)
    study_ann.optimize(objective, n_trials=10)
    optuna_ann_mse_score = study_ann.best_value
    optuna_ann_time = (study_ann.best_trial.datetime_complete -
                       study_ann.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_ann_time, 60)
    h, m = divmod(m, 60)
    optuna_ann_time = "%d:%02d:%09f" % (h, m, s)
    print("ANN MSE score:%.4f" % optuna_ann_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_ann_time))
    print("最佳超参数值集合:", study_ann.best_params)
    model_optuna_ann = ANN(**study_ann.best_params)
    save_model_object(model_optuna_ann, 'Optuna', 'ANN', 'ANN')
    return optuna_ann_mse_score, optuna_ann_time, study_ann.best_params


def optuna_xgb(X, y):
    # 参考 https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
    # https://data-analysis-stats.jp/%e6%a9%9f%e6%a2%b0%e5%ad%a6%e7%bf%92/python%e3%81%a7xgboost/
    def objective(trial):
        params = {
            'learning_rate':
            trial.suggest_float("learning_rate", 1e-4, 1, log=True),
            'max_depth':
            trial.suggest_int("max_depth", 1, 46, step=5),
            'n_estimators':
            trial.suggest_int("n_neighbors", 100, 220, step=30),
            'objective':
            'reg:squarederror',
        }
        clf = xgb.XGBRegressor(**params)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective, n_trials=10)
    optuna_xgb_mse_score = study_xgb.best_value
    optuna_xgb_time = (study_xgb.best_trial.datetime_complete -
                       study_xgb.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_xgb_time, 60)
    h, m = divmod(m, 60)
    optuna_xgb_time = "%d:%02d:%09f" % (h, m, s)
    print("XGBoost MSE score:%.4f" % optuna_xgb_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_xgb_time))
    print("最佳超参数值集合:", study_xgb.best_params)
    save_model_object(study_xgb, 'BO-TPE', 'NGBoost', 'NGBoost')
    return optuna_xgb_mse_score, optuna_xgb_time, study_xgb.best_params


def optuna_lightgbm(X, y):
    # 参考
    # https://github.com/optuna/optuna/blob/master/examples/lightgbm_simple.py
    # https://qiita.com/TomokIshii/items/3729c1b9c658cc48b5cb

    def objective(trial):
        #     下面这个是分类classification使用的模型，不能用在regressor
        #     dtrain = lgb.Dataset(X_train, label=y_train)

        params = {
            "objective":
            "regression",
            #         "metric": "mse",
            #         "verbose": 0,
            "boosting_type":
            "gbdt",
            "lambda_l1":
            trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
            "lambda_l2":
            trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
            "num_leaves":
            trial.suggest_int("num_leaves", 10, 100),
            "learning_rate":
            trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            #         "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            #         "learning_rate": trial.suggest_float("learning_rate",1e-4, 1, log=True),
            # "num_leaves": 30,
            # 'learning_rate': 0.1,
            'feature_fraction':
            0.9,
            'bagging_fraction':
            0.8,
            'bagging_freq':
            5,
            'verbose':
            -1,
            "min_child_samples":
            trial.suggest_int("min_child_samples", 5, 100),
            # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }

        clf = lgb.LGBMRegressor(**params)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))
        return score

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective, n_trials=10)
    optuna_lgb_mse_score = study_lgb.best_value
    optuna_lgb_time = (study_lgb.best_trial.datetime_complete -
                       study_lgb.best_trial.datetime_start).total_seconds()
    # 秒数转化为时间格式
    m, s = divmod(optuna_lgb_time, 60)
    h, m = divmod(m, 60)
    optuna_lgb_time = "%d:%02d:%09f" % (h, m, s)
    print("LightGBM MSE score:%.4f" % optuna_lgb_mse_score)
    print("程序执行时间（秒）:{}".format(optuna_lgb_time))
    print("最佳超参数值集合:", study_lgb.best_params)
    save_model_object(study_lgb, 'BO-TPE', 'NGBoost', 'NGBoost')
    return optuna_lgb_mse_score, optuna_lgb_time, study_lgb.best_params
