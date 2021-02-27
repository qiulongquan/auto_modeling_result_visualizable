#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
from pathlib import Path
# 获取绝对路径，然后加入到sys.path系统路径中去，这样系统运行的时候就可以找到自定义库了
sys.path.append(str(Path(__file__).parents[1]))
from src.basic_config import CHART_TITLE_1, CHART_TITLE_2, CHART_TITLE_3, CHART_FILE_NAME_1, CHART_FILE_NAME_2, CHART_FILE_NAME_3, \
    font_path, baseline, grid_search, random_search, BO_GP, gp_minimize, BO_TPE, Optuna, all_result_file_path
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载中文显示
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
# 显示中文或者日文的方法
# https://blog.csdn.net/hezuijiudexiaobai/article/details/104533154
# font_set = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",size=20)
font_set = fm.FontProperties(fname=font_path)


def collection_result_process(range_max):
    optimizer_list = [[baseline, 'base_score_df'],
                      [grid_search, 'gridsearch_score_df'],
                      [random_search, 'randomsearch_score_df'],
                      [BO_GP, 'bo_score_df'],
                      [gp_minimize, 'gp_minimize_score_df'],
                      [BO_TPE, 'BO_TPE_score_df'], [Optuna, 'Optuna_score_df']]
    dataframe_df = pd.DataFrame([])
    for i in range(range_max - 1):
        with open(
                os.path.join('model_and_record', optimizer_list[i][0],
                             optimizer_list[i][1] + '.pkl'), "rb") as f:
            # 読み出し
            optimizer_list[i][1] = pickle.load(f)
            if optimizer_list[i][1] is not None or len(
                    optimizer_list[i][1]) > 0:
                dataframe_df = pd.concat([dataframe_df, optimizer_list[i][1]])
        # print(dataframe_df)
    # 保存最终统计结果
    dataframe_df.to_csv(os.path.join('model_and_record', 'all_result.csv'))
    dataframe_df.to_pickle(os.path.join('model_and_record', 'all_result.pkl'))
    # 读取保存数据结构
    dataframe_df = None
    with open(all_result_file_path, "rb") as f:
        dataframe_df = pickle.load(f)
    # 显示分数统计图
    dataframe_df_score = dataframe_df.iloc[[0, 2, 4, 6, 8, 10, 12]]
    dataframe_df_score.reset_index(drop=False, inplace=True)
    # print(dataframe_df_score)

    # 数据格式转换
    type_df = pd.DataFrame(
        [['RandomForestRegressor'] * (range_max - 1),
         ['SVR'] * (range_max - 1), ['KNeighborsRegressor'] * (range_max - 1),
         ['ANN'] * (range_max - 1), ['NGBoost'] * (range_max - 1),
         ['GradientBoostingRegressor'] * (range_max - 1),
         ['XGboost'] * (range_max - 1), ['LightGBM'] * (range_max - 1)])
    type_df = type_df.values.reshape(-1, 1)
    type_df = pd.DataFrame(type_df)

    # 数据格式转换
    data_temp_svr = dataframe_df_score[['description', 'SVR']]
    data_temp_svr = data_temp_svr.rename(
        columns={'SVR': 'RandomForestRegressor'})
    data_temp_KNeighborsRegressor = dataframe_df_score[[
        'description', 'KNeighborsRegressor'
    ]]
    data_temp_KNeighborsRegressor = data_temp_KNeighborsRegressor.rename(
        columns={'KNeighborsRegressor': 'RandomForestRegressor'})
    data_temp_ANN = dataframe_df_score[['description', 'ANN']]
    data_temp_ANN = data_temp_ANN.rename(
        columns={'ANN': 'RandomForestRegressor'})
    data_temp_NGBoost = dataframe_df_score[['description', 'NGBoost']]
    data_temp_NGBoost = data_temp_NGBoost.rename(
        columns={'NGBoost': 'RandomForestRegressor'})
    data_temp_GradientBoostingRegressor = dataframe_df_score[[
        'description', 'GradientBoostingRegressor'
    ]]
    data_temp_GradientBoostingRegressor = data_temp_GradientBoostingRegressor.rename(
        columns={'GradientBoostingRegressor': 'RandomForestRegressor'})
    data_temp_XGboost = dataframe_df_score[['description', 'XGboost']]
    data_temp_XGboost = data_temp_XGboost.rename(
        columns={'XGboost': 'RandomForestRegressor'})
    data_temp_LightGBM = dataframe_df_score[['description', 'LightGBM']]
    data_temp_LightGBM = data_temp_LightGBM.rename(
        columns={'LightGBM': 'RandomForestRegressor'})
    transform_df = pd.concat([
        dataframe_df_score, data_temp_svr, data_temp_KNeighborsRegressor,
        data_temp_ANN, data_temp_NGBoost, data_temp_GradientBoostingRegressor,
        data_temp_XGboost, data_temp_LightGBM
    ],
                             axis=0)

    transform_df['model_category'] = type_df.values
    transform_df = transform_df.drop(columns=[
        'SVR', 'KNeighborsRegressor', 'ANN', 'NGBoost',
        'GradientBoostingRegressor', 'XGboost', 'LightGBM'
    ])
    transform_df = transform_df.rename(
        columns={'RandomForestRegressor': 'score'})
    # print(transform_df)

    # 有一些位置是np.nan所以需要填充0
    clean_z = transform_df['score'].fillna(0)
    # 有一些位置因为没有值所以需要补上0
    clean_z[clean_z == ''] = 0
    transform_df['score'] = clean_z
    # print(transform_df)
    # score列格式转换，转换成float
    transform_df['score'] = transform_df['score'].astype('float')
    # 整理后的数据，画图表示
    sns.set(style="whitegrid", color_codes=True)
    fig = plt.figure(figsize=(16, 8))
    sns.barplot(x="description",
                y="score",
                hue="model_category",
                data=transform_df)
    plt.grid(True)
    plt.title(CHART_TITLE_1, FontProperties=font_set)
    plt.legend(loc='upper left')
    # 保存图片到指定目录
    fig.savefig(CHART_FILE_NAME_1)
    plt.show()
    return transform_df


def dataframe_sort_show(transform_df):
    # ['score']列按照升序排列
    transform_df_sort = transform_df.sort_values(by=['score'], ascending=True)
    transform_df_sort.reset_index(drop=True, inplace=True)
    print(transform_df_sort)
    for index, _ in transform_df_sort.iterrows():
        if transform_df_sort.loc[index]['score'] == 0:
            transform_df_sort.drop([index], inplace=True)
    transform_df_sort.reset_index(drop=True, inplace=True)
    print(transform_df_sort)
    fig = plt.figure(figsize=(16, 12))
    sns.barplot(x="score",
                y="description",
                hue="model_category",
                data=transform_df_sort)
    plt.title(CHART_TITLE_2, FontProperties=font_set)
    fig.savefig(CHART_FILE_NAME_2)
    plt.show()

    for index, _ in transform_df_sort.iterrows():
        transform_df_sort.loc[
            index, 'description'] = "model:" + transform_df_sort.loc[index][
                'model_category'] + "  HyperOptimizer:" + transform_df_sort.loc[
                    index]['description'].rsplit('_', 1)[0]
    transform_df_sort.drop(['model_category'], axis=1, inplace=True)
    print(transform_df_sort)
    fig = plt.figure(figsize=(20, 12))
    sns.barplot(x="score", y="description", data=transform_df_sort)
    plt.title(CHART_TITLE_3, FontProperties=font_set)
    fig.savefig(CHART_FILE_NAME_3)
    plt.show()
