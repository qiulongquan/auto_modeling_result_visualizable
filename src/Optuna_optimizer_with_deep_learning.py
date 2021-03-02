#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Program Name：自动模型构筑并精度比较程序(基于深度学习构筑regression模型)
Created date：2020-03-02
Updated date：2020-03-02
Author：QIU
Objective：程序的主程序，基于深度学习构筑regression模型
"""

import sys
import os
from pathlib import Path
# 获取当前文件的父目录路径，然后加入到sys.path系统路径中去，这样系统运行的时候就可以找到自定义库了
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 导入数据X,y
from src.basic_config import X, y
from src.deep_learning_model import trainer
from src.common.get_logger_instance import log_output
from src.common.save_model_and_result_record import save_model_object, save_record_object, load_model_object, load_record_object
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import optuna
import time
import pickle
import datetime


# 基于深度学习自动构筑最优模型以及自动超参数优化
def objective(trial):
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    model = trainer(trial, x_train, y_train)
    evaluate = model.evaluate(x=x_valid, y=y_valid)
    print("evaluate=", evaluate)
    return evaluate[1]


def main():
    start_time = datetime.datetime.now()

    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    # 最优结果输出
    print("best_value:{}".format(study.best_value))
    print("best_params:{}".format(study.best_params))

    end_time = datetime.datetime.now()
    process_time = end_time - start_time
    print("程序执行总时间（秒）:{}".format(process_time))


if __name__ == '__main__':
    main()
