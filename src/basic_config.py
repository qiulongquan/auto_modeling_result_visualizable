#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import datasets
import os

# 图表的title文字定义
CHART_TITLE_1 = '在波士顿房价数据集，不同模型MSE测试分数(分数越低越好，可以看出RandomForest,LightGBM,NGBoost模型都是不错的选择)'
CHART_TITLE_2 = '在波士顿房价数据集，不同模型MSE测试分数(分数越低越好)'
CHART_TITLE_3 = '在波士顿房价数据集，不同模型MSE测试分数排序后结果(分数越低越好，可以看出NGBoost,LightGBM,RandomForest模型都是不错的选择)'

# 图表的file name定义
CHART_FILE_NAME_1 = "models_MSE_summary.png"
CHART_FILE_NAME_2 = "models_MSE_sort.png"
CHART_FILE_NAME_3 = "models_MSE_sort_1.png"

# 导入数据（默认是波士顿房价数据集）
X, y = datasets.load_boston(return_X_y=True)

# 模型使用的种类范围
RANGE_MIN = 1
RANGE_MAX = 8

# 模型保存的子目录定义
baseline = 'baseline'
grid_search = 'grid_search'
random_search = 'random_search'
BO_GP = 'BO-GP'
gp_minimize = 'gp_minimize'
BO_TPE = 'BO-TPE'
Optuna = 'Optuna'

# 中文字符显示字符集path定义
font_path = os.path.join(os.getcwd(), 'src', 'common', 'msyh.ttc')

# 汇总数据path定义
all_result_file_path = 'model_and_record/all_result.pkl'

# 神经模型列表
NN_model_list = ['ANN']
# 神经网络训练时是否输出训练内容
# 1表示显示，0表示不显示
verbose = 0
