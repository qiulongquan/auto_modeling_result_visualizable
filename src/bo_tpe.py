#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from src.common.save_model_and_result_record import save_model_object, save_record_object
from skopt import Optimizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope
from keras.wrappers.scikit_learn import KerasRegressor
from src.ANN_model import ANN
from src.basic_config import verbose
import datetime
import numpy as np

# NGBoost需要的包
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
from ngboost.scores import MLE
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings


def bo_tpe_RandomForestRegressor(X, y):

    starttime = datetime.datetime.now()

    # Define the objective function
    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'max_features': int(params['max_features']),
            "min_samples_split": int(params['min_samples_split']),
            "min_samples_leaf": int(params['min_samples_leaf']),
            "criterion": str(params['criterion'])
        }
        clf = RandomForestRegressor(**params)
        score = -np.mean(
            cross_val_score(
                clf, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))

        return {'loss': score, 'status': STATUS_OK}

    # Define the hyperparameter configuration space
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 150, 1),
        'max_depth': hp.quniform('max_depth', 5, 50, 1),
        "max_features": hp.quniform('max_features', 1, 13, 1),
        "min_samples_split": hp.quniform('min_samples_split', 2, 11, 1),
        "min_samples_leaf": hp.quniform('min_samples_leaf', 1, 11, 1),
        "criterion": hp.choice('criterion', ['mse', 'mae'])
    }
    trials_rf = Trials()
    best_rf = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=20,
                   trials=trials_rf)
    print("Random Forest MSE score:%.4f" % min(trials_rf.losses()))
    endtime = datetime.datetime.now()
    process_time_rf = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_rf))
    print("最佳超参数值集合:", best_rf)
    save_model_object(best_rf, 'BO-TPE', 'RandomForestRegressor',
                      'RandomForestRegressor')
    return min(trials_rf.losses()), process_time_rf, best_rf


def bo_tpe_svr(X, y):
    starttime = datetime.datetime.now()

    def objective(params):
        params = {
            'C': abs(float(params['C'])),
            "kernel": str(params['kernel']),
            'epsilon': abs(float(params['epsilon'])),
        }
        clf = SVR(gamma='scale', **params)
        score = -np.mean(
            cross_val_score(
                clf, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))

        return {'loss': score, 'status': STATUS_OK}

    space = {
        'C': hp.normal('C', 0, 50),
        "kernel": hp.choice('kernel', ['poly', 'rbf', 'sigmoid']),
        'epsilon': hp.normal('epsilon', 0, 1),
    }

    trials_svr = Trials()
    best_svr = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials_svr)
    print("SVM MSE score:%.4f" % min(trials_svr.losses()))
    endtime = datetime.datetime.now()
    process_time_svr = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_svr))
    print("最佳超参数值集合:", best_svr)
    save_model_object(best_svr, 'BO-TPE', 'SVR', 'SVR')
    return min(trials_svr.losses()), process_time_svr, best_svr


def bo_tpe_knn(X, y):
    starttime = datetime.datetime.now()

    def objective(params):
        params = {'n_neighbors': abs(int(params['n_neighbors']))}
        clf = KNeighborsRegressor(**params)
        score = -np.mean(
            cross_val_score(
                clf, X, y, cv=3, n_jobs=-1, scoring="neg_mean_squared_error"))
        return {'loss': score, 'status': STATUS_OK}

    space = {
        'n_neighbors': hp.quniform('n_neighbors', 1, 20, 1),
    }

    trials_knn = Trials()
    best_knn = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials_knn)
    print("KNN MSE score:%.4f" % min(trials_knn.losses()))
    endtime = datetime.datetime.now()
    process_time_knn = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_knn))
    print("最佳超参数值集合:", best_knn)
    save_model_object(best_knn, 'BO-TPE', 'KNN', 'KNN')
    return min(trials_knn.losses()), process_time_knn, best_knn


def bo_tpe_ANN(X, y):
    starttime = datetime.datetime.now()

    def objective(params):
        params = {
            "activation": str(params['activation']),
            "loss": str(params['loss']),
            'batch_size': abs(int(params['batch_size'])),
            'neurons': abs(int(params['neurons'])),
            'epochs': abs(int(params['epochs'])),
            'learning_rate': abs(float(params['learning_rate']))
        }
        clf = KerasRegressor(build_fn=ANN, **params, verbose=verbose)
        score = -np.mean(
            cross_val_score(clf, X, y, cv=3, scoring="neg_mean_squared_error"))

        return {'loss': score, 'status': STATUS_OK}

    space_activation = ['relu', 'tanh']
    space_loss = ['mse', 'mae']
    space = {
        "activation": hp.choice('activation', space_activation),
        "loss": hp.choice('loss', space_loss),
        'batch_size': hp.quniform('batch_size', 32, 128, 32),
        'neurons': hp.quniform('neurons', 256, 1024, 256),
        'epochs': hp.quniform('epochs', 30, 60, 10),
        'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-2)
    }

    trials_ann = Trials()
    best_ann = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials_ann)
    print("ANN MSE score:%.4f" % min(trials_ann.losses()))
    endtime = datetime.datetime.now()
    process_time_ann = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_ann))
    print("最佳超参数值集合:", best_ann)
    best_params_ann = {
        'activation': space_activation[best_ann['activation']],
        'loss': space_loss[best_ann['loss']],
        'batch_size': int(best_ann['batch_size']),
        'neurons': int(best_ann['neurons']),
        'epochs': int(best_ann['epochs']),
        'learning_rate': float(best_ann['learning_rate'])
    }
    model_bo_tpe_ann = ANN(**best_params_ann)
    save_model_object(model_bo_tpe_ann, 'BO-TPE', 'ANN', 'ANN')
    return min(trials_ann.losses()), process_time_ann, best_ann


def bo_tpe_ngb(X, y):
    # 参考例子
    # https://github.com/stanfordmlgroup/ngboost/blob/master/examples/tuning/hyperopt.ipynb
    data = X
    target = y
    # 2次数据划分，这样可以分成3份数据  test  train  validation
    X_intermediate, X_test, y_intermediate, y_test = train_test_split(
        data, target, shuffle=True, test_size=0.2, random_state=1)

    # train/validation split (gives us train and validation sets)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_intermediate,
        y_intermediate,
        shuffle=False,
        test_size=0.25,
        random_state=1)

    # delete intermediate variables
    del X_intermediate, y_intermediate

    # 显示数据集的分配比例
    print('train: {}% | validation: {}% | test {}%'.format(
        round((len(y_train) / len(target)) * 100, 2),
        round((len(y_validation) / len(target)) * 100, 2),
        round((len(y_test) / len(target)) * 100, 2)))

    starttime = datetime.datetime.now()

    # 搜索空间设定
    b1 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=2)
    b2 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
    b3 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)

    space = {
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
        'minibatch_frac': hp.choice('minibatch_frac', [1.0, 0.5]),
        'Base': hp.choice('Base', [b1, b2, b3])
    }

    # n_estimators表示一套参数下，有多少个评估器，简单说就是迭代多少次
    default_params = {"n_estimators": 20, "verbose_eval": 1, "random_state": 1}

    def objective(params):
        params.update(default_params)
        ngb = NGBRegressor(**params, verbose=False).fit(
            X_train,
            y_train,
            X_val=X_validation,
            Y_val=y_validation,
            #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
            early_stopping_rounds=2)
        loss = ngb.evals_result['val']['LOGSCORE'][ngb.best_val_loss_itr]
        results = {'loss': loss, 'status': STATUS_OK}
        return results

    trials_ngb = Trials()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            # max_evals是设定多少套参数组合，组合数越大准确度可能更高但是训练的时间越长
            max_evals=50,
            trials=trials_ngb)

    best_params = space_eval(space, best)

    ngb_new = NGBRegressor(**best_params, verbose=False).fit(
        X_train,
        y_train,
        X_val=X_validation,
        Y_val=y_validation,
        #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
        early_stopping_rounds=2)

    y_pred = ngb_new.predict(X_test)
    test_MSE_ngb = mean_squared_error(y_pred, y_test)
    print("NGBoost MSE score:%.4f" % test_MSE_ngb)
    endtime = datetime.datetime.now()
    process_time_ngb = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_ngb))
    print("最佳超参数值集合:", best_params)
    save_model_object(ngb_new, 'BO-TPE', 'NGBoost', 'NGBoost')
    return test_MSE_ngb, process_time_ngb, best_params


def bo_tpe_lightgbm(X, y):
    # 参考
    # https://qiita.com/TomokIshii/items/3729c1b9c658cc48b5cb

    data = X
    target = y
    # 2次数据划分，这样可以分成3份数据  test  train  validation
    X_intermediate, X_test, y_intermediate, y_test = train_test_split(
        data, target, shuffle=True, test_size=0.2, random_state=1)

    # train/validation split (gives us train and validation sets)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_intermediate,
        y_intermediate,
        shuffle=False,
        test_size=0.25,
        random_state=1)

    # delete intermediate variables
    del X_intermediate, y_intermediate

    # 显示数据集的分配比例
    print('train: {}% | validation: {}% | test {}%'.format(
        round((len(y_train) / len(target)) * 100, 2),
        round((len(y_validation) / len(target)) * 100, 2),
        round((len(y_test) / len(target)) * 100, 2)))

    starttime = datetime.datetime.now()

    space = {
        # 'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
        # 'minibatch_frac': hp.choice('minibatch_frac', [1.0, 0.5]),
        # 'Base': hp.choice('Base', [b1, b2, b3])
        "lambda_l1": hp.uniform("lambda_l1", 1e-8, 1.0),
        "lambda_l2": hp.uniform("lambda_l2", 1e-8, 1.0),
        "min_child_samples": hp.uniformint("min_child_samples", 5, 100),
        'learning_rate': hp.uniform("learning_rate", 0.001, 0.5),
        "n_estimators": hp.uniformint("n_estimators", 10, 100),
        "num_leaves": hp.uniformint("num_leaves", 5, 35)
    }

    # n_estimators表示一套参数下，有多少个评估器，简单说就是迭代多少次
    default_params = {
        # "n_estimators": 80,
        "random_state": 1,
        "objective": "regression",
        "boosting_type": "gbdt",
        # "num_leaves": 30,
        # "learning_rate": 0.3,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    def objective(params):
        #     下面这个是分类classification使用的模型，不能用在regressor
        #     dtrain = lgb.Dataset(X_train, label=y_train)
        params.update(default_params)
        clf = lgb.LGBMRegressor(**params)
        score = -np.mean(
            cross_val_score(clf,
                            X_train,
                            y_train,
                            cv=3,
                            n_jobs=-1,
                            scoring="neg_mean_squared_error"))
        return {'loss': score, 'status': STATUS_OK}

    trials_lgb = Trials()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            # max_evals是设定多少套参数组合，组合数越大准确度可能更高但是训练的时间越长
            max_evals=50,
            trials=trials_lgb)

    best_params = space_eval(space, best)
    lgb_model = lgb.LGBMRegressor(**best_params).fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=-1,
        #  假定n_estimators迭代器有100个设定了早期停止后也许不到100次迭代就完成了训练停止了
        early_stopping_rounds=2)

    y_pred = lgb_model.predict(X_test)
    test_MSE_lgb = mean_squared_error(y_pred, y_test)
    print("LightGBM MSE score:%.4f" % test_MSE_lgb)
    endtime = datetime.datetime.now()
    process_time_lgb = endtime - starttime
    print("程序执行时间（秒）:{}".format(process_time_lgb))
    print("最佳超参数值集合:", best_params)
    save_model_object(lgb_model, 'BO-TPE', 'NGBoost', 'NGBoost')
    return test_MSE_lgb, process_time_lgb, best_params
