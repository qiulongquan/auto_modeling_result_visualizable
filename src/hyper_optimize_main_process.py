#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Program Name：自动模型构筑并精度比较程序
Created date：2020-02-16
Updated date：2020-02-21
Author：QIU
Objective：程序的主程序，调用各个模型构筑子程序
"""

import sys
import os
from pathlib import Path
# 获取当前文件的父目录路径，然后加入到sys.path系统路径中去，这样系统运行的时候就可以找到自定义库了
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.basic_config import X, y, RANGE_MAX, RANGE_MIN
from src.common.get_logger_instance import log_output
from src.baseline import bs_random_forest_regressor, bs_svr, bs_KNN, bs_ANN
from src.grid_search import grid_knn, grid_RandomForestRegressor, grid_svr, grid_ANN
from src.random_search import rs_RandomForestRegressor, rs_svr, rs_knn, rs_ANN
from src.bo_gp import bo_RandomForestRegressor, bo_svr, bo_knn, bo_ANN
from src.gp_minimize import gpminimize_RandomForestRegressor, gpminimize_svr, gpminimize_knn
from src.bo_tpe import bo_tpe_knn, bo_tpe_RandomForestRegressor, bo_tpe_svr, bo_tpe_ANN, bo_tpe_ngb, bo_tpe_lightgbm
from src.optuna_optimizer import optuna_GradientBoostingRegressor, optuna_knn, optuna_lightgbm, optuna_RandomForestRegressor, optuna_svr, optuna_xgb,optuna_ANN
from src.common.save_model_and_result_record import save_model_object, save_record_object, load_model_object, load_record_object
from src.common.collection_result_process import collection_result_process, dataframe_sort_show
from time import sleep
from tqdm import tqdm
import pandas as pd
import time
import pickle


def baseline_model_group():
    # random_forest_regressor模型调用
    base_rf_score, process_time_rf = bs_random_forest_regressor(X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'baseline'
    message = "model_name: {}, hyper_optimize: {}, base_rf_score: {}, process_time_rf: {}".format(
        model_name, hyper_optimize, base_rf_score, process_time_rf)
    log_output(message)

    # svr模型调用
    base_svr_score, process_time_svr = bs_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'baseline'
    message = "model_name: {}, hyper_optimize: {}, base_svr_score: {}, process_time_svr: {}".format(
        model_name, hyper_optimize, base_svr_score, process_time_svr)
    log_output(message)

    # knn模型调用
    base_knn_score, process_time_knn = bs_KNN(X, y)
    model_name = 'KNN'
    hyper_optimize = 'baseline'
    message = "model_name: {}, hyper_optimize: {}, base_knn_score: {}, process_time_knn: {}".format(
        model_name, hyper_optimize, base_knn_score, process_time_knn)
    log_output(message)

    # ann模型调用
    base_ann_score, process_time_ann = bs_ANN(X, y)
    model_name = 'ANN'
    hyper_optimize = 'baseline'
    message = "model_name: {}, hyper_optimize: {}, base_ann_score: {}, process_time_ann: {}".format(
        model_name, hyper_optimize, base_ann_score, process_time_ann)
    log_output(message)

    score = [
        'baseline_score', base_rf_score, base_svr_score, base_knn_score,
        base_ann_score
    ]
    process_time = [
        'baseline_process_time', process_time_rf, process_time_svr,
        process_time_knn, process_time_ann
    ]
    base_score = [score, process_time]
    base_score_df = pd.DataFrame(data=base_score,
                                 columns=[
                                     'description', 'RandomForestRegressor',
                                     'SVR', 'KNeighborsRegressor', 'ANN'
                                 ])
    base_score_df.set_index('description', inplace=True)
    save_record_object(base_score_df, 'baseline', 'base_score_df')


def grid_search_model_group():
    # random_forest_regressor模型调用
    grid_rf_score, process_time_rf, best_params_rf = grid_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'grid_search'
    message = "model_name: {}, hyper_optimize: {}, grid_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, grid_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    grid_svr_score, process_time_svr, best_params_svr = grid_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'grid_search'
    message = "model_name: {}, hyper_optimize: {}, grid_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, grid_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    grid_knn_score, process_time_knn, best_params_knn = grid_knn(X, y)
    model_name = 'KNN'
    hyper_optimize = 'grid_search'
    message = "model_name: {}, hyper_optimize: {}, grid_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, grid_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    # ann模型调用
    grid_ann_score, process_time_ann, best_params_ann = grid_ANN(X, y)
    model_name = 'ANN'
    hyper_optimize = 'grid_search'
    message = "model_name: {}, hyper_optimize: {}, grid_ann_score: {}, process_time_ann: {},best_params_ann:{}".format(
        model_name, hyper_optimize, grid_ann_score, process_time_ann,
        best_params_ann)
    log_output(message)

    score = [
        'gridsearch_score', grid_rf_score, grid_svr_score, grid_knn_score,
        grid_ann_score
    ]
    process_time = [
        'gridsearch_process_time', process_time_rf, process_time_svr,
        process_time_knn, process_time_ann
    ]
    gridsearch_score = [score, process_time]
    gridsearch_score_df = pd.DataFrame(data=gridsearch_score,
                                       columns=[
                                           'description',
                                           'RandomForestRegressor', 'SVR',
                                           'KNeighborsRegressor', 'ANN'
                                       ])
    gridsearch_score_df.set_index('description', inplace=True)
    save_record_object(gridsearch_score_df, 'grid_search',
                       'gridsearch_score_df')


def random_search_model_group():
    # random_forest_regressor模型调用
    random_rf_score, process_time_rf, best_params_rf = rs_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'random_search'
    message = "model_name: {}, hyper_optimize: {}, random_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, random_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    random_svr_score, process_time_svr, best_params_svr = rs_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'random_search'
    message = "model_name: {}, hyper_optimize: {}, random_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, random_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    random_knn_score, process_time_knn, best_params_knn = rs_knn(X, y)
    model_name = 'KNN'
    hyper_optimize = 'random_search'
    message = "model_name: {}, hyper_optimize: {}, random_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, random_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    # ann模型调用
    random_ann_score, process_time_ann, best_params_ann = rs_ANN(X, y)
    model_name = 'ANN'
    hyper_optimize = 'random_search'
    message = "model_name: {}, hyper_optimize: {}, random_ann_score: {}, process_time_ann: {},best_params_ann:{}".format(
        model_name, hyper_optimize, random_ann_score, process_time_ann,
        best_params_ann)
    log_output(message)

    score = [
        'randomsearch_score', random_rf_score, random_svr_score,
        random_knn_score, random_ann_score
    ]
    process_time = [
        'randomsearch_process_time', process_time_rf, process_time_svr,
        process_time_knn, process_time_ann
    ]
    randomsearch_score = [score, process_time]
    randomsearch_score_df = pd.DataFrame(data=randomsearch_score,
                                         columns=[
                                             'description',
                                             'RandomForestRegressor', 'SVR',
                                             'KNeighborsRegressor', 'ANN'
                                         ])
    randomsearch_score_df.set_index('description', inplace=True)
    save_record_object(randomsearch_score_df, 'random_search',
                       'randomsearch_score_df')


def bo_gp_model_group():
    # random_forest_regressor模型调用
    bo_rf_score, process_time_rf, best_params_rf = bo_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'BO-GP'
    message = "model_name: {}, hyper_optimize: {}, bo_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, bo_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    bo_svr_score, process_time_svr, best_params_svr = bo_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'BO-GP'
    message = "model_name: {}, hyper_optimize: {}, bo_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, bo_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    bo_knn_score, process_time_knn, best_params_knn = bo_knn(X, y)
    model_name = 'KNN'
    hyper_optimize = 'BO-GP'
    message = "model_name: {}, hyper_optimize: {}, bo_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, bo_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    # ann模型调用
    bo_ann_score, process_time_ann, best_params_ann = bo_ANN(X, y)
    model_name = 'ANN'
    hyper_optimize = 'BO-GP'
    message = "model_name: {}, hyper_optimize: {}, bo_ann_score: {}, process_time_ann: {},best_params_ann:{}".format(
        model_name, hyper_optimize, bo_ann_score, process_time_ann,
        best_params_ann)
    log_output(message)

    score = ['bo_score', bo_rf_score, bo_svr_score, bo_knn_score, bo_ann_score]
    process_time = [
        'bo_process_time', process_time_rf, process_time_svr, process_time_knn,
        process_time_ann
    ]
    bo_score = [score, process_time]
    bo_score_df = pd.DataFrame(data=bo_score,
                               columns=[
                                   'description', 'RandomForestRegressor',
                                   'SVR', 'KNeighborsRegressor', 'ANN'
                               ])
    bo_score_df.set_index('description', inplace=True)
    save_record_object(bo_score_df, 'BO-GP', 'bo_score_df')


def gp_minimize_model_group():
    # random_forest_regressor模型调用
    gpminimize_rf_score, process_time_rf, best_params_rf = gpminimize_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'gp_minimize'
    message = "model_name: {}, hyper_optimize: {}, gpminimize_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, gpminimize_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    gpminimize_svr_score, process_time_svr, best_params_svr = gpminimize_svr(
        X, y)
    model_name = 'SVR'
    hyper_optimize = 'gp_minimize'
    message = "model_name: {}, hyper_optimize: {}, gpminimize_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, gpminimize_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    gpminimize_knn_score, process_time_knn, best_params_knn = gpminimize_knn(
        X, y)
    model_name = 'KNN'
    hyper_optimize = 'gp_minimize'
    message = "model_name: {}, hyper_optimize: {}, gpminimize_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, gpminimize_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    score = [
        'gp_minimize_score', gpminimize_rf_score, gpminimize_svr_score,
        gpminimize_knn_score, 0
    ]
    process_time = [
        'gp_minimize_process_time', process_time_rf, process_time_svr,
        process_time_knn, 0
    ]
    gp_minimize_score = [score, process_time]
    gp_minimize_score_df = pd.DataFrame(data=gp_minimize_score,
                                        columns=[
                                            'description',
                                            'RandomForestRegressor', 'SVR',
                                            'KNeighborsRegressor', 'ANN'
                                        ])
    gp_minimize_score_df.set_index('description', inplace=True)
    save_record_object(gp_minimize_score_df, 'gp_minimize',
                       'gp_minimize_score_df')


def bo_tpe_model_group():
    # random_forest_regressor模型调用
    bo_tpe_rf_score, process_time_rf, best_params_rf = bo_tpe_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, bo_tpe_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    bo_tpe_svr_score, process_time_svr, best_params_svr = bo_tpe_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, bo_tpe_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    bo_tpe_knn_score, process_time_knn, best_params_knn = bo_tpe_knn(X, y)
    model_name = 'KNN'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, bo_tpe_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    # ann模型调用
    bo_tpe_ann_score, process_time_ann, best_params_ann = bo_tpe_ANN(X, y)
    model_name = 'ANN'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_ann_score: {}, process_time_ann: {},best_params_ann:{}".format(
        model_name, hyper_optimize, bo_tpe_ann_score, process_time_ann,
        best_params_ann)
    log_output(message)

    # ngb模型调用
    bo_tpe_ngb_score, process_time_ngb, best_params_ngb = bo_tpe_ngb(X, y)
    model_name = 'NGBoost'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_ngb_score: {}, process_time_ngb: {},best_params_ngb:{}".format(
        model_name, hyper_optimize, bo_tpe_ngb_score, process_time_ngb,
        best_params_ngb)
    log_output(message)

    # lgb模型调用
    bo_tpe_lgb_score, process_time_lgb, best_params_lgb = bo_tpe_lightgbm(X, y)
    model_name = 'LightGBM'
    hyper_optimize = 'BO-TPE'
    message = "model_name: {}, hyper_optimize: {}, bo_tpe_gmb_score: {}, process_time_gbm: {},best_params_lgb:{}".format(
        model_name, hyper_optimize, bo_tpe_lgb_score, process_time_lgb,
        best_params_lgb)
    log_output(message)

    score = [
        'BO-TPE_score', bo_tpe_rf_score, bo_tpe_svr_score, bo_tpe_knn_score,
        bo_tpe_ann_score, bo_tpe_ngb_score, bo_tpe_lgb_score
    ]
    process_time = [
        'BO-TPE_process_time', process_time_rf, process_time_svr,
        process_time_knn, process_time_ann, process_time_ngb, process_time_lgb
    ]
    BO_TPE_score = [score, process_time]
    BO_TPE_score_df = pd.DataFrame(data=BO_TPE_score,
                                   columns=[
                                       'description', 'RandomForestRegressor',
                                       'SVR', 'KNeighborsRegressor', 'ANN',
                                       'NGBoost', 'LightGBM'
                                   ])
    BO_TPE_score_df.set_index('description', inplace=True)
    save_record_object(BO_TPE_score_df, 'BO-TPE', 'BO_TPE_score_df')


def optuna_model_group():
    # random_forest_regressor模型调用
    optuna_rf_score, process_time_rf, best_params_rf = optuna_RandomForestRegressor(
        X, y)
    model_name = 'random_forest_regressor'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_rf_score: {}, process_time_rf: {},best_params_rf:{}".format(
        model_name, hyper_optimize, optuna_rf_score, process_time_rf,
        best_params_rf)
    log_output(message)

    # svr模型调用
    optuna_svr_score, process_time_svr, best_params_svr = optuna_svr(X, y)
    model_name = 'SVR'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_svr_score: {}, process_time_svr: {},best_params_svr:{}".format(
        model_name, hyper_optimize, optuna_svr_score, process_time_svr,
        best_params_svr)
    log_output(message)

    # knn模型调用
    optuna_knn_score, process_time_knn, best_params_knn = optuna_knn(X, y)
    model_name = 'KNN'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_knn_score: {}, process_time_knn: {},best_params_knn:{}".format(
        model_name, hyper_optimize, optuna_knn_score, process_time_knn,
        best_params_knn)
    log_output(message)

    # # ann模型调用
    # optuna_ann_score, process_time_ann,best_params_ann = optuna_ANN(X, y)
    # model_name = 'ANN'
    # hyper_optimize = 'Optuna'
    # message = "model_name: {}, hyper_optimize: {}, optuna_ann_score: {}, process_time_ann: {},best_params_ann:{}".format(
    #     model_name, hyper_optimize, optuna_ann_score, process_time_ann,best_params_ann)
    # log_output(message)

    # GradientBoostingRegressor模型调用
    optuna_gbr_score, process_time_gbr, best_params_gbr = optuna_GradientBoostingRegressor(
        X, y)
    model_name = 'GradientBoostingRegressor'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_gbr_score: {}, process_time_gbr: {},best_params_gbr:{}".format(
        model_name, hyper_optimize, optuna_gbr_score, process_time_gbr,
        best_params_gbr)
    log_output(message)

    # xgb模型调用
    optuna_xgb_score, process_time_xgb, best_params_xgb = optuna_xgb(X, y)
    model_name = 'XGBoost'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_xgb_score: {}, process_time_xgb: {},best_params_xgb:{}".format(
        model_name, hyper_optimize, optuna_xgb_score, process_time_xgb,
        best_params_xgb)
    log_output(message)

    # lightgbm模型调用
    optuna_lgb_score, process_time_lgb, best_params_lgb = optuna_lightgbm(X, y)
    model_name = 'LightGBM'
    hyper_optimize = 'Optuna'
    message = "model_name: {}, hyper_optimize: {}, optuna_lgb_score: {}, process_time_lgb: {},best_params_lgb:{}".format(
        model_name, hyper_optimize, optuna_lgb_score, process_time_lgb,
        best_params_lgb)
    log_output(message)

    score = [
        'Optuna_score', optuna_rf_score, optuna_gbr_score, 0, optuna_svr_score,
        optuna_knn_score, optuna_xgb_score, optuna_lgb_score
    ]

    process_time = [
        'Optuna_process_time', process_time_rf, process_time_gbr, 0,
        process_time_svr, process_time_knn, process_time_xgb, process_time_lgb
    ]
    Optuna_score = [score, process_time]
    Optuna_score_df = pd.DataFrame(data=Optuna_score,
                                   columns=[
                                       'description', 'RandomForestRegressor',
                                       'GradientBoostingRegressor', 'ANN',
                                       'SVR', 'KNeighborsRegressor', 'XGboost',
                                       'LightGBM'
                                   ])
    Optuna_score_df.set_index('description', inplace=True)
    save_record_object(Optuna_score_df, 'Optuna', 'Optuna_score_df')


if __name__ == '__main__':
    if (len(sys.argv) > 1) and (sys.argv[1] == "debug"):
        import ptvsd
        print("waiting...")
        ptvsd.enable_attach(address=("0.0.0.0", 3000))
        ptvsd.wait_for_attach()
    print("data shape check:", X.shape, y.shape)
    for i in tqdm(range(RANGE_MIN, RANGE_MAX)):
        if i == 1:
            print("\n-----------------baseline-----------------")
            baseline_model_group()
            base_score_df = load_record_object('baseline', 'base_score_df.pkl')
            print(base_score_df)
        elif i == 2:
            print(
                "\n-----------------grid search hyperOptimize-----------------"
            )
            grid_search_model_group()
            gridsearch_score_df = load_record_object(
                'grid_search', 'gridsearch_score_df.pkl')
            print(gridsearch_score_df)
        elif i == 3:
            print(
                "\n-----------------random search hyperOptimize-----------------"
            )
            random_search_model_group()
            randomsearch_score_df = load_record_object(
                'random_search', 'randomsearch_score_df.pkl')
            print(randomsearch_score_df)
        elif i == 4:
            print(
                "\n-----------------bayesian optimizer with Gaussian Process hyperOptimize-----------------"
            )
            bo_gp_model_group()
            bo_score_df = load_record_object('BO-GP', 'bo_score_df.pkl')
            print(bo_score_df)
        elif i == 5:
            print(
                "\n-----------------Gaussian Process minimize hyperOptimize-----------------"
            )
            gp_minimize_model_group()
            gp_minimize_score_df = load_record_object(
                'gp_minimize', 'gp_minimize_score_df.pkl')
            print(gp_minimize_score_df)
        elif i == 6:
            print(
                "\n-----------------Bayesian Optimization with Tree-structured Parzen Estimator hyperOptimize-----------------"
            )
            bo_tpe_model_group()
            BO_TPE_score_df = load_record_object('BO-TPE',
                                                 'BO_TPE_score_df.pkl')
            print(BO_TPE_score_df)
        elif i == 7:
            print("\n-----------------Optuna hyperOptimize-----------------")
            optuna_model_group()
            Optuna_score_df = load_record_object('Optuna',
                                                 'Optuna_score_df.pkl')
            print(Optuna_score_df)

    print(
        "\n====================================================数据汇总处理开始===================================================="
    )
    # 收集所有保存的MSE分数统计结果，并汇总,然后显示图表
    transform_df = collection_result_process(RANGE_MAX)
    # MSE分数排序并显示最优模型和优化方式图表
    dataframe_sort_show(transform_df)
